import json
import random
import argparse
from tqdm import tqdm
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import numpy as np
from sklearn.metrics import precision_recall_curve, accuracy_score, precision_score, recall_score, f1_score
import wandb
from transformers import TrainingArguments
from transformers import Trainer, EarlyStoppingCallback
import os
from transformers import TrainerCallback

from utis.judge_include import search_ngram
from utis.mask_data import mask_data, mask_data_some, mask_data_multi
from utis.data import create_datasets, CustomDataCollator
from utis.model import HalNPMTrainModel as HalNPMModel
from utis.loss import compute_loss_train


random.seed(42)

# 実行コマンド: python train_npm.py --pretrain --small --mask_mode multi --save_name 1226_modernbert_pretrain --epochs 5

def main():
    parser = argparse.ArgumentParser(description="NPM Train")
    parser.add_argument("--pretrain", action="store_true", help="continue pretraining")
    parser.add_argument("--onlyhal", action="store_true", help="only hallucinated data")
    parser.add_argument("--cut_faith", action="store_true", help="cut faithful data")
    parser.add_argument("--small", action="store_true", help="small data")
    parser.add_argument("--perfect", action="store_true", help="perfect match")
    parser.add_argument("--mask_mode", type=str, default="single", choices=["single", "some", "multi"], help="mask mode")
    parser.add_argument("--mask_ratio", type=float, default=0.15, help="mask ratio for some spans")
    parser.add_argument("--p", type=float, default=0.5, help="p for geometric distribution in multi mask mode")
    parser.add_argument("--save_name", type=str, default="npm_model", help="model save name")
    parser.add_argument("--epochs", type=int, default=10, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="learning rate")
    parser.add_argument("--base_model", type=str, default="answerdotai/ModernBERT-large", help="base model name")
    parser.add_argument("--loss_mode", type=str, default="contrastive", choices=["contrastive", "margin"], help="loss function mode")
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="disable Weights & Biases logging (for dry run / debugging)",
    )


    args = parser.parse_args()

    if_pretrain = args.pretrain
    if_onlyhal = args.onlyhal
    if_cut_faith = args.cut_faith
    if_small = args.small
    if_perfect = args.perfect
    mask_mode = args.mask_mode
    mask_ratio = args.mask_ratio
    p = args.p
    name = args.save_name
    num_epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    base_model = args.base_model
    loss_mode = args.loss_mode
    use_wandb = not args.no_wandb

    if use_wandb:
        os.environ["WANDB_PROJECT"] = "huggingface"
        wandb.init(
            name=args.save_name,
            config=vars(args),
        )
    else:
        os.environ["WANDB_DISABLED"] = "true"
    
    # これらのハイパラをprint
    print(f"Pretrain: {if_pretrain}")
    print(f"Only hallucinated data: {if_onlyhal}")
    print(f"Cut faithful data: {if_cut_faith}")
    print(f"Small data: {if_small}")
    print(f"Perfect match: {if_perfect}")
    print(f"Mask mode: {mask_mode}")
    if mask_mode == "some":
        print(f"Mask ratio: {mask_ratio}")
    if mask_mode == "multi":
        print(f"Mask ratio: {mask_ratio}")
        print(f"p (geometric distribution): {p}")
    print(f"Save name: {name}")
    print(f"Num epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Base model: {base_model}")
    print(f"Loss mode: {loss_mode}")

    with open(f"data/1127_srl_train.jsonl", "r") as f:
        train_data = [json.loads(line) for line in f]
    with open("data/1127_srl_dev.jsonl", "r") as f:
        dev_data = [json.loads(line) for line in f]
    with open("data/1127_srl_test.jsonl", "r") as f:
        test_data = [json.loads(line) for line in f]
    
    if if_pretrain:
        # faithfulだけ
        train_data = [d for d in train_data if d["labels"]==[]]
        dev_data = [d for d in dev_data if d["labels"]==[]]
        test_data = [d for d in test_data if d["labels"]==[]]
    if if_onlyhal: 
        # hallucinatedだけ
        train_data = [d for d in train_data if d["labels"]!=[]]
        dev_data = [d for d in dev_data if d["labels"]!=[]]
        test_data = [d for d in test_data if d["labels"]!=[]]
    if if_cut_faith:
        # faithfulを半分に削減
        faith_train = [d for d in train_data if d["labels"]==[]]
        hal_train = [d for d in train_data if d["labels"]!=[]]
        faith_dev = [d for d in dev_data if d["labels"]==[]]
        hal_dev = [d for d in dev_data if d["labels"]!=[]]
        faith_test = [d for d in test_data if d["labels"]==[]]
        hal_test = [d for d in test_data if d["labels"]!=[]]
        train_data = hal_train + random.sample(faith_train, len(faith_train)//2)
        dev_data = hal_dev + random.sample(faith_dev, len(faith_dev)//2)
        test_data = hal_test + random.sample(faith_test, len(faith_test)//2)
        
    if if_small:
        train_data = random.sample(train_data, 800)
        #train_data = [d for d in train_data if d.get("source_id") == "15595"][0:1]
        dev_data = random.sample(dev_data, 200)
        test_data = random.sample(test_data, 200)
    
    train_data = search_ngram(train_data, perfect=if_perfect)
    dev_data = search_ngram(dev_data, perfect=if_perfect)
    test_data = search_ngram(test_data, perfect=if_perfect)

    #tokenizer = AutoTokenizer.from_pretrained("facebook/npm")
    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-large", use_fast=True)
    if mask_mode == "single":
        train_data = mask_data(train_data, tokenizer)
        dev_data = mask_data(dev_data, tokenizer)
        test_data = mask_data(test_data, tokenizer)
    elif mask_mode == "some":
        train_data = mask_data_some(train_data, tokenizer, mask_ratio, if_onlyhal)
        dev_data = mask_data_some(dev_data, tokenizer, mask_ratio, if_onlyhal)
        test_data = mask_data_some(test_data, tokenizer, mask_ratio, if_onlyhal)
    elif mask_mode == "multi":
        train_data = mask_data_multi(train_data, tokenizer, mask_ratio, p)
        dev_data = mask_data_multi(dev_data, tokenizer, mask_ratio, p)
        test_data = mask_data_multi(test_data, tokenizer, mask_ratio, p)
    #train_data = [train_data[0]]
    print(f"Train data size: {len(train_data)}")
    print(f"Dev data size: {len(dev_data)}")
    print(f"Test data size: {len(test_data)}")

    tokenized_datasets = create_datasets(train_data, dev_data, test_data, tokenizer)
    data_collator = CustomDataCollator(tokenizer=tokenizer)

    if base_model == "facebook/npm":
        model = AutoModel.from_pretrained("facebook/npm")
    elif base_model == "answerdotai/ModernBERT-large":
        model = AutoModel.from_pretrained("answerdotai/ModernBERT-large")
    else: # 自分で訓練したモデル
        model = HalNPMModel.from_pretrained(
            base_model=AutoModel.from_pretrained("answerdotai/ModernBERT-large"),
            loss_function=compute_loss_train,
            tokenizer=tokenizer,
            top_k=5,
            save_directory=base_model,
        ).base_model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    training_args = TrainingArguments(
        output_dir="./results",
        report_to=(["wandb"] if use_wandb else []),
        run_name=(name if use_wandb else None),
        eval_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=32,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        fp16=True,
        gradient_accumulation_steps=2,
        logging_dir="./logs",
        remove_unused_columns=False,
        optim="adafactor",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )


    hal_model = HalNPMModel(base_model=model, loss_function=compute_loss_train, tokenizer=tokenizer,loss_mode=loss_mode)


    callbacks = [EarlyStoppingCallback(early_stopping_patience=2)]

    trainer = Trainer(
        hal_model,
        training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["dev"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    #trainer.evaluate()
    trainer.train()
    
    save_name =f"trained_model/{name}"
    trainer.save_model(save_name)
    trainer.save_state()
    hal_model.save_pretrained(save_name)
    

def compute_metrics(eval_preds):
    preds_raw, labels_raw = eval_preds
    preds_raw = np.asarray(preds_raw)
    preds_flat = preds_raw.ravel()
    labels_raw = np.asarray(labels_raw)
    labels_flat = labels_raw.ravel()
    labels = labels_flat[np.isin(labels_flat, [0,1])]
    preds = preds_flat[np.isin(labels_flat, [0,1])]
    # preds をサンプル単位の 1D スコア配列に展開する
    if isinstance(preds, list):
        parts = []
        for p in preds:
            a = np.asarray(p)
            if a.ndim == 0:
                parts.append(np.array([float(a)]))
            elif a.ndim == 1:
                parts.append(a.astype(float))
            else:
                # (batch_size, dim) の場合は各サンプルごとに代表値を取る（max）
                parts.append(np.max(a, axis=1).astype(float))
        scores = np.concatenate(parts)
    else:
        scores = np.asarray(preds)
        if scores.ndim > 1:
            # (n_steps, batch) などの2Dは flatten（またはサンプル軸が先頭なら axis=0 を flatten）
            if scores.size == labels.size:
                scores = scores.ravel()
            elif scores.shape[0] == labels.size:
                scores = np.max(scores, axis=1)
            elif scores.shape[1] == labels.size:
                scores = np.max(scores, axis=0)
            else:
                scores = scores.ravel()

    scores = np.asarray(scores, dtype=float).ravel()

    if scores.size != labels.size:
        raise ValueError(f"preds length {scores.size} is incompatible with labels length {labels.size}")

    # PR 曲線で F1 最大の閾値を選ぶ
    try:
        precision_vals, recall_vals, thresholds = precision_recall_curve(labels, -scores)
        if thresholds.size == 0:
            best_threshold = 0.0
        else:
            f1_scores = 2 * (precision_vals[:-1] * recall_vals[:-1]) / (precision_vals[:-1] + recall_vals[:-1] + 1e-12)
            best_idx = int(np.argmax(f1_scores))
            best_threshold = -1.0 * float(thresholds[best_idx])
    except Exception:
        best_threshold = 0.0

    preds_bin = (scores <= best_threshold).astype(int)
    return {
        "best_threshold": best_threshold,
        "accuracy": float(accuracy_score(labels, preds_bin)),
        "precision": float(precision_score(labels, preds_bin, zero_division=0)),
        "recall": float(recall_score(labels, preds_bin, zero_division=0)),
        "f1": float(f1_score(labels, preds_bin, zero_division=0)),
    }
    
if __name__ == "__main__":
    main()
