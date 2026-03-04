import re
import numpy as np
import torch
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import json
from transformers import AutoTokenizer, AutoModelForTokenClassification
from tqdm import tqdm
from sklearn.metrics import precision_recall_curve, f1_score


# -----------------------------
# Config
# -----------------------------
@dataclass
class InferConfig:
    model_name_or_path: str
    max_length: int = 8192
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------------
# Utility: token probs -> char spans in text
# -----------------------------
def probs_to_char_spans(
    probs: np.ndarray,
    offsets: List[Tuple[int, int]],
    threshold: float = 0.5,
) -> List[Tuple[int, int]]:
    """
    probs: shape [S] (text単独tokenの P(hall=1))
    offsets: text単独tokenの offset_mapping (list of (start,end) in text chars)
    threshold: hall判定のしきい値
    returns: [(start_char, end_char), ...] merged spans (end exclusive)
    """
    assert len(probs) == len(offsets)
    spans = []
    in_span = False
    cur_s, cur_e = None, None

    for p, (st, ed) in zip(probs, offsets):
        if st == ed:  # 空offset（念のため）
            continue
        is_h = p >= threshold
        if is_h and not in_span:
            in_span = True
            cur_s, cur_e = st, ed
        elif is_h and in_span:
            # 連結（隣接/重なりは結合）
            if st <= cur_e:
                cur_e = max(cur_e, ed)
            else:
                spans.append((cur_s, cur_e))
                cur_s, cur_e = st, ed
        elif (not is_h) and in_span:
            spans.append((cur_s, cur_e))
            in_span = False
            cur_s, cur_e = None, None

    if in_span:
        spans.append((cur_s, cur_e))

    return spans


def normalize_spaces(s: str) -> str:
    # decodeで出る余計なスペース調整用（必要なら）
    return re.sub(r"\s+", " ", s).strip()


# -----------------------------
# Core: sliding-window inference with max aggregation
# -----------------------------
@torch.no_grad()
def predict_text_token_probs(
    model: AutoModelForTokenClassification,
    tokenizer: AutoTokenizer,
    document: str,
    text: str,
    cfg: InferConfig,
) -> Dict[str, Any]:

    model.eval()
    model.to(cfg.device)

    enc = tokenizer(
        document,
        text,
        truncation=True,  # 両方まとめて長さ制限（必要なら）
        max_length=cfg.max_length,
        return_offsets_mapping=True,
        padding=False,
    )

    input_ids = torch.tensor(enc["input_ids"], device=cfg.device).unsqueeze(0)
    attention_mask = torch.tensor(enc["attention_mask"], device=cfg.device).unsqueeze(0)

    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits  # [1, L, C]
    logits = logits[0]  # [L, C]

    probs_all = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()  # [L]

    # text側だけ拾う
    # fast tokenizer なら sequence_ids が取れる
    seq_ids = enc.sequence_ids()  # list length L; 0=document, 1=text, None=special
    offsets = enc["offset_mapping"]  # list[(st,ed)] length L

    probs_text = []
    offsets_text = []
    for sid, (st, ed), pr in zip(seq_ids, offsets, probs_all):
        if sid != 1:
            continue
        if st == ed == 0:
            continue
        probs_text.append(float(pr))
        offsets_text.append((int(st), int(ed)))

    # token文字列も欲しいなら（text側部分だけ）
    # 注意: convert_ids_to_tokensはspecialも含むので、同様にsid==1だけ取る
    toks_all = tokenizer.convert_ids_to_tokens(enc["input_ids"])
    token_strs_text = [
        tok for tok, sid, (st, ed) in zip(toks_all, seq_ids, offsets) if sid == 1 and not (st == ed == 0)
    ]

    return {
        "skipped_reason": None,
        "probs": np.array(probs_text, dtype=np.float32),
        "token_offsets": offsets_text,
        "token_strs": token_strs_text,
    }


# -----------------------------
# High-level helper: inference + spans
# -----------------------------
def infer_hallucination_spans(
    model: AutoModelForTokenClassification,
    tokenizer: AutoTokenizer,
    document: str,
    text: str,
    cfg: InferConfig,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Returns:
      - skipped_reason (or None)
      - token_probs (or None)
      - spans_char: merged hallucination spans in text char indices
      - spans_text: the extracted substrings (for convenience)
    """
    out = predict_text_token_probs(model, tokenizer, document, text, cfg)
    if out is None or out.get("skipped_reason") is not None:
        return {
            "skipped_reason": out.get("skipped_reason") if out else "unknown",
            "token_probs": None,
            "spans_char": [],
            "spans_text": [],
        }

    probs = out["probs"]
    offsets = out["token_offsets"]

    spans_char = probs_to_char_spans(probs, offsets, threshold=threshold)
    spans_text = [text[s:e] for (s, e) in spans_char]

    return {
        "skipped_reason": None,
        "token_probs": probs,
        "spans_char": spans_char,
        "spans_text": spans_text,
        "token_strs": out["token_strs"],
        "token_offsets": offsets,
    }


def eval_char(item):
    hal_label = []
    for d in item["labels"]:
        hal_label.extend(list(range(d["start"], d["end"])))
    hal_label = set(hal_label)

    hal_pred = []
    for sp in item["preds"]:
        hal_pred.extend(list(range(sp["start"], sp["end"])))
    hal_pred = set(hal_pred)
    tp = len(hal_label & hal_pred)
    fp = len(hal_pred - hal_label)
    fn = len(hal_label - hal_pred)
    # print(hal_label, hal_pred)
    return tp, fp, fn


def compute_metrics(results):
    tp = fp = fn = 0
    for item in results:
        tp_i, fp_i, fn_i = eval_char(item)
        tp += tp_i
        fp += fp_i
        fn += fn_i
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
    }


# -----------------------------
# Example usage
# -----------------------------
def load_model_and_tokenizer(cfg: InferConfig):
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(cfg.model_name_or_path)
    return model, tokenizer


if __name__ == "__main__":
    cfg = InferConfig(
        model_name_or_path="trained_model/0107_modernbert_baseline_binary",
        max_length=8192,
    )

    output_file = "modernbert_inference_results/0107_modernbert_binary.jsonl"
    model, tokenizer = load_model_and_tokenizer(cfg)

    with open("data/1127_srl_test.jsonl", "r") as f:
        test_data = [json.loads(line) for line in f]
    results = []

    for sample in tqdm(test_data):
        document = sample["source_info"] if type(sample["source_info"]) == str else sample["source_info"]["passages"]
        text = sample["response"]

        res = infer_hallucination_spans(
            model=model,
            tokenizer=tokenizer,
            document=document,
            text=text,
            cfg=cfg,
            threshold=0.5,
        )

        if res["skipped_reason"]:
            print("SKIPPED:", res["skipped_reason"])
        # else:
        # print(res)
        # print("Spans (char):", res["spans_char"])
        # print("Spans (text):", res["spans_text"])
        # tokenごとの確率を見たいなら：
        # for tok, p, (s,e) in zip(res["token_strs"], res["token_probs"], res["token_offsets"]):
        #     print(tok, float(p), (s,e))
        pred = []
        for s, e in res["spans_char"]:
            pred.append(
                {
                    "start": s,
                    "end": e,
                    "text": text[s:e],
                }
            )
        preds_all = []
        if res.get("token_probs") is not None:
            for p, off in zip(res["token_probs"], res["token_offsets"]):
                preds_all.append(
                    {
                        "text": text[off[0] : off[1]],
                        "prob": float(p),
                        "start": off[0],
                        "end": off[1],
                    }
                )

        results.append(
            {
                "id_name": sample["id_name"],
                "source_id": sample["source_id"],
                "task_type": sample["task_type"],
                "document": document,
                "text": text,
                "labels": sample["labels"],
                "preds": pred,
                "preds_all": preds_all,
            }
        )
    with open(output_file, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(compute_metrics(results))
    print(compute_metrics([d for d in results if d["task_type"] == "QA"]))
    print(compute_metrics([d for d in results if d["task_type"] == "Summary"]))

    hallucinated_data = [d for d in results if d["labels"] != []]
    print("Hallucinated data metrics:")
    print(compute_metrics(hallucinated_data))
    print(compute_metrics([d for d in hallucinated_data if d["task_type"] == "QA"]))
    print(compute_metrics([d for d in hallucinated_data if d["task_type"] == "Summary"]))
