# infer_lora.py（最小）
import argparse, json
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
from tqdm import tqdm


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="meta-llama/Llama-2-13b-hf")
    ap.add_argument("--lora_dir", required=True)
    ap.add_argument("--in_jsonl", default="data/ft_test.jsonl")
    ap.add_argument("--out", default="ft_inference_results/1027_preds_lora_llama2.jsonl")
    ap.add_argument("--max_new", type=int, default=256)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.base_model, device_map="auto", torch_dtype="auto")
    model = PeftModel.from_pretrained(model, args.lora_dir)

    with open(args.in_jsonl, "r", encoding="utf-8") as f, open(args.out, "w", encoding="utf-8") as wf:
        for line in tqdm(f, desc="Processing"):
            ex = json.loads(line)
            prompt = ex["input_text"]
            inps = tok(prompt, return_tensors="pt").to(model.device)
            out = model.generate(
                **inps,
                do_sample=False,
                max_new_tokens=args.max_new,
                eos_token_id=tok.eos_token_id,
                pad_token_id=tok.eos_token_id
            )
            text = tok.decode(out[0], skip_special_tokens=True)
            gen = text[len(prompt) :]
            wf.write(json.dumps({"id": ex.get("id_name"), "raw": gen}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
