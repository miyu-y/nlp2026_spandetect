from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModel
import torch.nn as nn
import os
import torch.nn.functional as F
import torch
import numpy as np
import json
from tqdm import tqdm
import random
import math
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
import argparse
from typing import List, Tuple, Optional

from utis.mask_data import mask_data_inference as mask_data
from utis.data import create_datasets_inference as create_dataset
from utis.model import HalNPMInferenceModel as HalNPMModel
from utis.loss import contrastive_span_scores as contrastive_loss

parser = argparse.ArgumentParser(description="NPM inference")

parser.add_argument("--mode", choices=["dev", "test", "cls"], default="test", help="inference mode")
parser.add_argument("--full_data", action="store_true", help="full data")
parser.add_argument(
    "--model_name", dest="model_name", type=str, default="answerdotai/ModernBERT-large", help="model name"
)
parser.add_argument("--add_word", dest="add_word", type=str, default="", help="additional word for model name")
parser.add_argument("--small", action="store_true", help="small data")
parser.add_argument("--predword", action="store_true", help="show predicted span")


# 実行時コマンド: python inference_npm_ft_batch.py --model_name 1225_modernbert --predword --small

random.seed(42)
args = parser.parse_args()
mode = args.mode
full_data = args.full_data
model_name = args.model_name
add_word = args.add_word
if add_word != "":
    add_word = "_" + add_word
if_small = args.small
show_predword = args.predword

top_k = 5
batch_size = 32

s = f"_{mode}" if mode != "test" else ""
date_directory = model_name.split("_")[0] if "_" in model_name else ""
os.makedirs(f"npm_inference_results/{date_directory}", exist_ok=True)
output_file = f"npm_inference_results/{date_directory}/{model_name.replace('/', '_')}{'_predword' if show_predword else ''}{add_word}{s}{'_mini' if if_small else ''}{'_full' if full_data else ''}.jsonl"
model_path = (
    f"trained_model/{model_name}" if model_name != "answerdotai/ModernBERT-large" else "answerdotai/ModernBERT-large"
)

# calc_sim = True  # sim も計算するかどうか
###########
with open(f"data/1127_srl_{mode}{'' if full_data else '_hal'}.jsonl", "r") as f:
    test_data_raw = [json.loads(line) for line in f]
    # test_data_raw = test_data_raw[:10]
if if_small:
    test_data_raw = random.sample(test_data_raw, 2)
    # test_data_raw = test_data_raw[:1]

span_num_list = [len(d["srl_splits"]) for d in test_data_raw]
tokenizer = AutoTokenizer.from_pretrained(model_path)

test_data = mask_data(test_data_raw, tokenizer)

tokenized_datasets = create_dataset(test_data, tokenizer)
if model_name == "facebook/npm" or model_name == "answerdotai/ModernBERT-large":
    print(f"Using pretrained {model_name} model for inference.")
    model = HalNPMModel(
        base_model=AutoModel.from_pretrained(model_name),
        loss_function=contrastive_loss,
        tokenizer=tokenizer,
    )
    model.top_k = top_k
else:
    print(f"Using fine-tuned model from {model_path} for inference.")
    model = HalNPMModel.from_pretrained(
        # base_model=AutoModel.from_pretrained("facebook/npm"),
        base_model=AutoModel.from_pretrained("answerdotai/ModernBERT-large"),
        loss_function=contrastive_loss,
        tokenizer=tokenizer,
        top_k=top_k,
        save_directory=model_path,
    )

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()  

original_hidden_cache = {}


@torch.no_grad()
def get_original_hidden(original_input_ids, original_attention_mask):
    """
    original_input_ids: 1D LongTensor (L,)
    return: hidden (L,H) on CPU
    """
    key = tuple(original_input_ids.tolist())  # hashable
    if key in original_hidden_cache:
        return original_hidden_cache[key]

    inp = original_input_ids.unsqueeze(0).to(device)
    att = original_attention_mask.unsqueeze(0).to(device)

    out = model.base_model(input_ids=inp, attention_mask=att, return_dict=True)[0]  # (1,L,H)
    out = out[0].detach().cpu()  # (L,H)
    original_hidden_cache[key] = out
    return out


pred_scores = []
pred_sims = []
pred_span_ids = []

ids = tokenized_datasets["test"]
N = len(ids)

for i in tqdm(range(0, N, batch_size)):
    idxs = list(range(i, min(i + batch_size, N)))
    batch = [ids[j] for j in idxs]

    doc_tensors = [torch.tensor(b["doc_input_ids"], dtype=torch.long) for b in batch]
    doc_masks = [torch.tensor(b["doc_attention_mask"], dtype=torch.long) for b in batch]
    text_tensors = [torch.tensor(b["text_input_ids"], dtype=torch.long) for b in batch]
    text_masks = [torch.tensor(b["text_attention_mask"], dtype=torch.long) for b in batch]
    # masked_span_tensors = [torch.tensor(b["original_input_ids"], dtype=torch.long) for b in batch]
    # masked_span_masks   = [torch.tensor(b["original_attention_mask"], dtype=torch.long) for b in batch]
    # masked_span_indexes = [b["masked_span_index"] for b in batch]
    masked_span_outputs = []
    for b in batch:
        orig_ids = torch.tensor(b["original_input_ids"], dtype=torch.long)
        orig_msk = torch.tensor(b["original_attention_mask"], dtype=torch.long)
        (s, e) = b["masked_span_index"]  # (start, end) で e は exclusive の想定

        hidden = get_original_hidden(orig_ids, orig_msk)  # (L,H) CPU

        span_vec = hidden[s:e].mean(dim=0)  # (H,)

        masked_span_outputs.append(span_vec)

    masked_span_output_batch = torch.stack(masked_span_outputs, dim=0).to(device)

    batch_doc_input_ids = pad_sequence(doc_tensors, batch_first=True, padding_value=tokenizer.pad_token_id).to(device)
    batch_doc_attention_mask = pad_sequence(doc_masks, batch_first=True, padding_value=0).to(device)
    batch_text_input_ids = pad_sequence(text_tensors, batch_first=True, padding_value=tokenizer.pad_token_id).to(device)
    batch_text_attention_mask = pad_sequence(text_masks, batch_first=True, padding_value=0).to(device)
    # batch_masked_span_input_ids = pad_sequence(masked_span_tensors, batch_first=True, padding_value=tokenizer.pad_token_id).to(device)
    # batch_masked_span_attention_mask = pad_sequence(masked_span_masks, batch_first=True, padding_value=0).to(device)

    # sentence_ids（※ slidingだと “doc token列” とズレやすいので注意）
    sentence_ids = [b["sentence_index"] for b in batch]

    # ngram start/end
    ngram_starts = [b.get("ngram_token_start", None) for b in batch]
    ngram_ends = [b.get("ngram_token_end", None) for b in batch]

    with torch.no_grad():
        # ★モデル側 forward も sliding対応してる前提：
        #   doc_input_ids が (B,C,L) で来る
        out = model(
            # input_ids=[batch_doc_input_ids, batch_text_input_ids, batch_masked_span_input_ids],
            # attention_mask=[batch_doc_attention_mask, batch_text_attention_mask, batch_masked_span_attention_mask],
            input_ids=[batch_doc_input_ids, batch_text_input_ids],
            attention_mask=[batch_doc_attention_mask, batch_text_attention_mask],
            ngram_token_start=ngram_starts,
            ngram_token_end=ngram_ends,
            labels=None,
            # masked_span_index=masked_span_indexes,
            masked_span_output=masked_span_output_batch,
            sentence_ids=sentence_ids,
            return_dict=True,
        )

        loss, scores, sims, span_ids = out.to_tuple()

    pred_scores.extend(scores.detach().cpu().numpy().tolist())
    pred_sims.extend(sims.detach().cpu().numpy().tolist())
    pred_span_ids.extend(span_ids)  # span_ids は list のまま

result_list = []

for n in range(len(test_data_raw)):
    res = test_data_raw[n]["srl_splits"]
    for i in range(span_num_list[n]):
        score_list = pred_scores[sum(span_num_list[:n]) + i]
        sim_list = pred_sims[sum(span_num_list[:n]) + i]
        res[i]["top_k_scores"] = [float(s) for s in score_list]
        res[i]["top_k_sims"] = [float(s) for s in sim_list]
        res[i]["max_score"] = float(score_list[0])
        res[i]["max_sim"] = float(sorted(sim_list, reverse=True)[0])
        # 予測文字列
        if show_predword:
            span_token_idx_list = pred_span_ids[sum(span_num_list[:n]) + i]
            pred_spans = []
            for span_token_idx in span_token_idx_list:
                tokens = tokenizer.convert_ids_to_tokens(ids[sum(span_num_list[:n]) + i]["doc_input_ids"])
                try:
                    span_tokens = [tokens[idx] for idx in span_token_idx]
                except IndexError:
                    span_tokens = ["[ERROR]"]
                    raise IndexError(
                        f"IndexError: span_token_idx {span_token_idx} out of range for tokens of length {len(tokens)}"
                    )
                # トークンを文字列に変換
                span_text = tokenizer.convert_tokens_to_string(span_tokens)
                pred_spans.append(span_text)
            res[i]["predicted_spans"] = pred_spans

    result_list.append(
        {
            "id_name": test_data_raw[n]["id_name"],
            "source_id": test_data_raw[n]["source_id"],
            "task_type": test_data_raw[n]["task_type"],
            "model": test_data_raw[n]["model"],
            "source": test_data_raw[n]["source"],
            "source_info": test_data_raw[n]["source_info"],
            "response": test_data_raw[n]["response"],
            "labels": test_data_raw[n]["labels"],
            "split": test_data_raw[n]["split"],
            "spans": res,
        }
    )

with open(output_file, "w") as f:
    for item in result_list:
        f.write(json.dumps(item) + "\n")
