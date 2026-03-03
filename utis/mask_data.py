import random
import numpy as np

random.seed(42)

def mask_text(input_text, span_dict, tokenizer):
    """
    span_dict: split_spans で作った要素（"start","end" を使用）
    """
    s, e = span_dict["start"], span_dict["end"]
    mask_seq = tokenizer.mask_token * 2
    if s > 0 and input_text[s - 1] != " ":
        mask_seq = " " + mask_seq
    if e < len(input_text) and input_text[e] != " ":
        mask_seq = mask_seq + " "
    return input_text[:s] + mask_seq + input_text[e:]

def mask_text_multi(input_text, span_dict, tokenizer):
    """
    span_dict: split_spans で作った要素（"start","end" を使用）のリスト
    """
    for span in sorted(span_dict, key=lambda x: x["start"], reverse=True): # 後ろからmask
        s, e = span["start"], span["end"]
        mask_seq = tokenizer.mask_token * 2
        if s > 0 and input_text[s - 1] != " ":
            mask_seq = " " + mask_seq
        if e < len(input_text) and input_text[e] != " ":
            mask_seq = mask_seq + " "
        input_text = input_text[:s] + mask_seq + input_text[e:]
    return input_text

def mask_data(data, tokenizer):
    for d in data:
        if d["labels"]==[]:
            cand_spans = [s for s in d["srl_splits"] if s["included"]]
            mask_span = random.choice(cand_spans) if cand_spans else None
        else:
            cand_spans = [s for s in d["srl_splits"] if s["hallucinated"] and s["included"]]
            if not cand_spans:
                cand_spans = [s for s in d["srl_splits"] if s["included"]]
            mask_span = random.choice(cand_spans) if cand_spans else None
        if mask_span:
            d["response"] = mask_text(d["response"], mask_span, tokenizer)
            d["masked_span"] = [mask_span]
            d["labels"] = [1 if mask_span["hallucinated"] else 0]
    return data

def mask_data_some(data, tokenizer, mask_ratio, onlyhal):
    p=0.5
    masked_data = []
    for d in data:
        if d["labels"]==[]:
            cand_spans = [s for s in d["srl_splits"] if s["included"]]
        else:
            cand_spans = [s for s in d["srl_splits"] if s["hallucinated"] and s["included"]]
            if not cand_spans and not onlyhal:
                cand_spans = [s for s in d["srl_splits"] if s["included"]]
        mask_num = min(len(cand_spans), max(1, int(len(d["srl_splits"])*mask_ratio)))
        # 幾何分布でmask spanのtoken lenを決定
        mask_spans = []
        cand_spans_local = cand_spans[:]  # 破壊しないようにコピー
        while len(mask_spans) < mask_num and cand_spans_local:
            # 幾何分布から「欲しい token 長」をサンプル
            L = int(np.random.geometric(p))

            # 候補の中で一番近い token_len を持つ span を選ぶ
            diffs = []
            for s in cand_spans_local:
                tok_len = s["token_span"][1] - s["token_span"][0]
                diffs.append(abs(tok_len - L))

            min_diff = min(diffs)
            tied = [s for s, diff in zip(cand_spans_local, diffs) if diff == min_diff]

            span = random.choice(tied)  # ★同点はランダム
            mask_spans.append(span)
            cand_spans_local.remove(span)

        for mask_span in mask_spans:
            d_copy = d.copy()
            d_copy["response"] = mask_text(d["response"], mask_span, tokenizer)
            d_copy["masked_span"] = [mask_span]
            d_copy["labels"] = [1 if mask_span["hallucinated"] else 0]
            masked_data.append(d_copy)
            
    return masked_data

def mask_data_multi(data, tokenizer, mask_ratio, p):
    for d in data:
        if d["labels"]==[]:
            cand_spans = [s for s in d["srl_splits"] if s["included"]]
        else:
            cand_spans = [s for s in d["srl_splits"] if s["hallucinated"] and s["included"]]
            if not cand_spans:
                cand_spans = [s for s in d["srl_splits"] if s["included"]]
        mask_num = min(len(cand_spans), max(1, int(len(d["srl_splits"])*mask_ratio)))
        # 幾何分布でmask spanのtoken lenを決定
        mask_spans = []
        cand_spans_local = cand_spans[:]  # 破壊しないようにコピー

        while len(mask_spans) < mask_num and cand_spans_local:
            # 幾何分布から「欲しい token 長」をサンプル
            L = int(np.random.geometric(p))

            # 候補の中で一番近い token_len を持つ span を選ぶ
            diffs = []
            for s in cand_spans_local:
                tok_len = s["token_span"][1] - s["token_span"][0]
                diffs.append(abs(tok_len - L))

            min_diff = min(diffs)
            tied = [s for s, diff in zip(cand_spans_local, diffs) if diff == min_diff]

            span = random.choice(tied)  # ★同点はランダム
            mask_spans.append(span)
            cand_spans_local.remove(span)

        if mask_spans:
            mask_spans = sorted(mask_spans, key=lambda x: x["start"])
            # 実際にresponseをマスク
            d["response"] = mask_text_multi(d["response"], mask_spans, tokenizer)
            d["masked_span"] = mask_spans
            # spanごとにラベル（hallucinatedなら1, それ以外0）をつける
            d["labels"] = [1 if s.get("hallucinated", False) else 0 for s in mask_spans]

    return data

def mask_data_inference(data, tokenizer):
    masked_data = []
    for d in data:
        for s in d["srl_splits"]:
            masked_text = mask_text(d["response"], s, tokenizer)
            masked_data.append(
                {
                    "masked_text": masked_text,
                    "source_info": d["source_info"]["passages"] if type(d["source_info"]) == dict else d["source_info"],
                    #"masked_span": s["text"],
                    "original_text": d["response"],
                    "masked_span_index": (s["start"], s["end"]),
                    "sentence_index": d["sentence_ids"],
                }
            )
    return masked_data