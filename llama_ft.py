import argparse, os, json
from typing import List, Dict, Any
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
import wandb

IGNORE_IDX = -100


# ==== Data utils ====
def to_target_json(label_objs: List[Dict[str, Any]]) -> str:
    # labels: list[dict]（start/end/text/...）→ {"hallucination list":[...]} のJSON文字列へ
    spans = []
    if isinstance(label_objs, list):
        for lab in sorted(label_objs, key=lambda x: x.get("start", 0)):
            t = str(lab.get("text", "")).strip()
            if t:
                spans.append(t)
    return json.dumps({"hallucination list": spans}, ensure_ascii=False)


def load_jsonl_dataset(train_path, val_path):
    def gen(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                ex = json.loads(line)
                yield {"prompt": ex["input_text"], "target": to_target_json(ex.get("labels", []))}

    return {"train": list(gen(train_path)), "validation": list(gen(val_path))}


# ==== Collator ====
class DataCollatorCausalJson:
    """input = prompt + target(JSON)。labels は target 部分のみ、prompt 部分は -100。"""

    def __init__(self, tokenizer):
        self.tok = tokenizer

    def __call__(self, batch: List[Dict[str, Any]]):
        prompts = [b["prompt"] for b in batch]
        targets = [b["target"] for b in batch]  # すでにJSON文字列
        inputs = [p + t for p, t in zip(prompts, targets)]

        enc = self.tok(
            inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.tok.model_max_length,
        )
        labels = enc["input_ids"].clone()
        labels[:] = IGNORE_IDX

        # 末尾から target 長分だけ損失をかける
        for i, tgt in enumerate(targets):
            tgt_ids = self.tok(tgt, add_special_tokens=False)["input_ids"]
            tgt_len = min(len(tgt_ids), enc["input_ids"].shape[1])  # 末尾トリムの安全策
            labels[i, -tgt_len:] = enc["input_ids"][i, -tgt_len:]

        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"], "labels": labels}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="meta-llama/Llama-2-13b-hf")  # or Llama-2-13B
    ap.add_argument("--train_jsonl", default="data/ft_train.jsonl")
    ap.add_argument("--val_jsonl", default="data/ft_dev.jsonl")
    ap.add_argument("--out_dir", default="ft_results/1027_llama2_lora")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--r", type=int, default=8)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--bits", type=int, default=4, choices=[4, 8, 16])  # 4=QLoRA
    ap.add_argument("--bsz", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--max_len", type=int, default=4096)
    ap.add_argument("--eval_steps", type=int, default=100)
    ap.add_argument("--save_steps", type=int, default=100)
    ap.add_argument("--bf16", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    os.environ["WANDB_PROJECT"] = "huggingface"  # 好きなproject名に
    wandb.init(
        name=args.save_name,  # run_name と揃えるなら
        config=vars(args),  # 実行引数を全部保存
    )

    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.model_max_length = args.max_len

    data = load_jsonl_dataset(args.train_jsonl, args.val_jsonl)
    collator = DataCollatorCausalJson(tok)

    load_kwargs = dict(device_map="auto")
    if args.bits in (4, 8):
        load_kwargs.update(dict(load_in_4bit=(args.bits == 4), load_in_8bit=(args.bits == 8)))

    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype="auto", **load_kwargs)
    if args.bits in (4, 8):
        model = prepare_model_for_kbit_training(model)

    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    lora_cfg = LoraConfig(
        r=args.r,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)

    training_args = TrainingArguments(
        output_dir=args.out_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.bsz,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        warmup_ratio=0.03,
        logging_steps=10,
        save_total_limit=2,
        bf16=args.bf16,
        fp16=not args.bf16,
        gradient_checkpointing=True,
        report_to=["wandb"],
        remove_unused_columns=False,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",  # ロス最小を基準に
        greater_is_better=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=data["train"],
        eval_dataset=data["validation"],
        data_collator=collator,
        tokenizer=tok,
    )

    trainer.train()
    trainer.save_state()
    trainer.save_model(args.out_dir)


if __name__ == "__main__":
    main()
