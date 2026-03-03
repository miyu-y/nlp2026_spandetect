import numpy as np
from transformers import AutoTokenizer
import json
from datasets import Dataset, DatasetDict
import pandas as pd
import torch
from transformers import AutoModelForTokenClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from transformers import DataCollatorForTokenClassification


def main():
    MODEL_NAME = "answerdotai/ModernBERT-large"
    save_name = "0107_modernbert_baseline_binary"

    with open("data/1127_srl_train.jsonl", "r") as f:
        train_data = [json.loads(line) for line in f]
    with open("data/1127_srl_dev.jsonl", "r") as f:
        dev_data = [json.loads(line) for line in f]
    with open("data/1127_srl_test.jsonl", "r") as f:
        test_data = [json.loads(line) for line in f]
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    raw_datasets = DatasetDict(
        {
            "train": create_dataset(train_data),
            "dev": create_dataset(dev_data),
            "test": create_dataset(test_data),
        }
    )
    print(len(raw_datasets["train"]), len(raw_datasets["dev"]), len(raw_datasets["test"]))

    tokenized_datasets = raw_datasets.map(
        tokenize_batched,
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
        fn_kwargs={"tokenizer": tokenizer},
    )
    print("train rows:", tokenized_datasets["train"].num_rows)
    print("dev rows:", tokenized_datasets["dev"].num_rows)
    print("columns:", tokenized_datasets["train"].column_names)

    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer,
        padding=True,  # 動的padding
    )

    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    args = TrainingArguments(
        output_dir="./results",
        report_to=["wandb"],
        run_name=save_name,
        learning_rate=1e-6,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=2,
        num_train_epochs=5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        fp16=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["dev"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_binary,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    trainer.evaluate()
    trainer.train()
    trainer.save_model(f"trained_model/{save_name}")
    trainer.save_state()
    model.save_pretrained(f"trained_model/{save_name}")


def create_dataset(data):
    dataset = []
    for d in data:
        labels = []
        for l in d["labels"]:
            labels.append((l["start"], l["end"]))
        dataset.append(
            {
                "document": d["source_info"]["passages"] if type(d["source_info"]) == dict else d["source_info"],
                "text": d["response"],
                "labels": labels,
            }
        )
    dataset = Dataset.from_pandas(pd.DataFrame(dataset))
    return dataset


def make_binary_labels_from_char_spans(offsets, seq_ids, hall_spans):
    """
    offsets: offset_mapping for a single chunk: list[(start,end)]
    seq_ids: sequence_ids for a single chunk: list[None/0/1]
    hall_spans: [(s,e), ...] in text (summary) char indices, end exclusive
               may be None, [], [[], ...] etc.
    """
    labels = np.full(len(offsets), -100, dtype=np.int64)

    # まず text 側を 0 で埋める（hall情報が無い/空でも安全）
    for i, ((st, ed), sid) in enumerate(zip(offsets, seq_ids)):
        if sid == 1 and not (st == ed == 0):
            labels[i] = 0

    if not hall_spans:
        return labels

    spans = []
    for se in hall_spans:
        if not isinstance(se, (list, tuple)) or len(se) != 2:
            continue
        s, e = se
        try:
            s, e = int(s), int(e)
        except (TypeError, ValueError):
            continue
        if e > s:
            spans.append((s, e))

    if not spans:
        return labels

    for i, ((st, ed), sid) in enumerate(zip(offsets, seq_ids)):
        if sid != 1 or (st == ed == 0):
            continue
        for hs, he in spans:
            if not (ed <= hs or he <= st):  # overlap
                labels[i] = 1
                break

    return labels


def tokenize_batched(batch, tokenizer=None):
    """
    1事例 = 1入力（slidingなし）
    returns: {"input_ids": [...], "attention_mask": [...], "labels": [...]}
    """
    out = {"input_ids": [], "attention_mask": [], "labels": []}

    docs = batch["document"]
    texts = batch["text"]
    spans_list = batch["labels"]  # hallucination char spans in text (summary)

    # まとめてtokenize（pair）
    enc = tokenizer(
        docs,
        texts,
        truncation=True,  # ★長すぎる場合だけ切る（ModernBERTのmaxが不明なら安全側で残す）
        return_offsets_mapping=True,  # ★ラベル生成に必要
        padding=False,  # ★collatorでpad
    )

    # enc は batch で返るので i で取り出す
    for i in range(len(enc["input_ids"])):
        seq_ids = enc.sequence_ids(i)  # None / 0(doc) / 1(text)
        offsets = enc["offset_mapping"][i]  # list[(st,ed)]

        hall_spans = spans_list[i]

        lab = make_binary_labels_from_char_spans(
            offsets=offsets,
            seq_ids=seq_ids,
            hall_spans=hall_spans,
        )

        out["input_ids"].append(enc["input_ids"][i])
        out["attention_mask"].append(enc["attention_mask"][i])
        out["labels"].append(lab.tolist())

    return out


def compute_metrics_binary(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    mask = labels != -100
    y_true = labels[mask]
    y_pred = preds[mask]

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    prec = tp / (tp + fp + 1e-9)
    rec = tp / (tp + fn + 1e-9)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    return {"precision": prec, "recall": rec, "f1": f1, "pos_tokens": int((y_true == 1).sum())}


if __name__ == "__main__":
    main()
