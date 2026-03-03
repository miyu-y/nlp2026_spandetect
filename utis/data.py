from datasets import Dataset, DatasetDict
import pandas as pd
from transformers import DataCollatorWithPadding
from torch.nn.utils.rnn import pad_sequence
import torch
import numpy as np


def create_raw_dataset(data):
    dataset = []
    for d in data:
        masked_span = d.get("masked_span", None)
        if masked_span is None:
            continue
        dataset.append(
            {
                "document": d["source_info"]["passages"] if type(d["source_info"]) == dict else d["source_info"],
                "text": d["response"],  # masked
                "masked_span": [s["text"] for s in masked_span],
                "ngram_start": [s["ngram_start"] for s in masked_span],  # list[list]
                "ngram_end": [s["ngram_end"] for s in masked_span],
                "labels": d["labels"],
            }
        )
    dataset = Dataset.from_pandas(pd.DataFrame(dataset))
    return dataset
    
def tokenize_function(examples, tokenizer, max_length=8192):
    docs = examples["document"]
    texts = examples["text"]

    # 期待する形:
    # starts[i] = [ [s_1_1, ..., s_1_m1], [s_2_1, ..., s_2_m2], ..., [s_k_1, ..., s_k_mk] ]
    # ends[i]   = 同様
    starts = examples.get("ngram_start", None)
    ends = examples.get("ngram_end", None)
    labels = examples.get("labels", [None] * len(docs))

    # document tokenize（offset必要）
    doc_enc = tokenizer(
        docs,
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
        padding=False,
    )

    def _char_to_token_span(offsets, start_char, end_char):
        """1つの (start_char, end_char) -> (start_token_idx, end_token_idx) に変換"""
        if start_char is None or end_char is None:
            return None, None

        ts = None
        te = None

        # offsets: [(o0,o1), ...] special tokenは(0,0)が多い
        for t_idx, (o0, o1) in enumerate(offsets):
            if o0 == 0 and o1 == 0:
                continue

            if ts is None and (o0 <= start_char < o1):
                ts = t_idx

            # end_char は exclusive 前提： o1 >= end_char になった最初の token
            if te is None and (o1 >= end_char):
                te = t_idx
                break

        # fallback（オーバーラップで救済）
        if ts is None:
            for t_idx, (o0, o1) in enumerate(offsets):
                if o0 == 0 and o1 == 0:
                    continue
                if o0 < end_char and o1 > start_char:
                    ts = t_idx
                    break

        if te is None:
            for t_idx in range(len(offsets) - 1, -1, -1):
                o0, o1 = offsets[t_idx]
                if o0 == 0 and o1 == 0:
                    continue
                if o0 < end_char and o1 > start_char:
                    te = t_idx
                    break

        return ts, te

    ngram_token_starts = []
    ngram_token_ends = []

    for i, offsets in enumerate(doc_enc["offset_mapping"]):
        # starts/ends が無いケースは空にする（学習側でゼロ埋め等）
        if starts is None or ends is None or starts[i] is None or ends[i] is None:
            ngram_token_starts.append([])
            ngram_token_ends.append([])
            continue

        # starts[i] は「mask箇所 k 個」のリスト
        tok_starts_i = []
        tok_ends_i = []

        k = min(len(starts[i]), len(ends[i]))
        for l in range(k):
            # ここが「候補 m_l 個」のリスト
            start_list = starts[i][l] or []
            end_list = ends[i][l] or []

            m = min(len(start_list), len(end_list))
            tok_s_l = []
            tok_e_l = []

            for j in range(m):
                ts, te = _char_to_token_span(offsets, start_list[j], end_list[j])
                tok_s_l.append(ts)
                tok_e_l.append(te)

            tok_starts_i.append(tok_s_l)
            tok_ends_i.append(tok_e_l)

        ngram_token_starts.append(tok_starts_i)
        ngram_token_ends.append(tok_ends_i)

    # offset_mapping不要
    doc_enc.pop("offset_mapping", None)

    # text tokenize（multi-mask済みtextが来る想定）
    text_enc = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding=False,
    )

    return {
        "doc_input_ids": doc_enc["input_ids"],
        "doc_attention_mask": doc_enc["attention_mask"],
        "text_input_ids": text_enc["input_ids"],
        "text_attention_mask": text_enc["attention_mask"],
        # ★ 事例×mask箇所×候補数 の三重構造を保持
        "ngram_token_start": ngram_token_starts,
        "ngram_token_end": ngram_token_ends,
        "labels": labels,
    }

def create_datasets(train_data, dev_data, test_data, tokenizer):
    train_dataset = create_raw_dataset(train_data)
    dev_dataset = create_raw_dataset(dev_data)
    test_dataset = create_raw_dataset(test_data)

    raw_datasets = DatasetDict(
        {
            "train": train_dataset,
            "dev": dev_dataset,
            "test": test_dataset,
        }
    )
    n_faithful = sum(1 for d in raw_datasets["train"] if d["labels"][0] == 0)
    n_hallucinated = sum(1 for d in raw_datasets["train"] if d["labels"][0] == 1)
    print(f"Faithful Num: {n_faithful}, Hallucinated Num: {n_hallucinated}")

    tokenized_datasets = raw_datasets.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(
        ["document", "text", "masked_span", "ngram_start", "ngram_end"]
    )
    return tokenized_datasets

class CustomDataCollator(DataCollatorWithPadding):
    def __init__(self, tokenizer, pad_labels_value: int = -100):
        super().__init__(tokenizer=tokenizer)
        self.tokenizer = tokenizer
        self.pad_labels_value = pad_labels_value

    def _count_masks(self, ids):
        if isinstance(ids, list):
            return int((np.array(ids) == self.tokenizer.mask_token_id).sum())
        t = ids if isinstance(ids, torch.Tensor) else torch.tensor(ids)
        return int((t == self.tokenizer.mask_token_id).sum().item())

    def __call__(self, features):
        # 1) まず全サンプルを “残す” 前提で k を推定し、無効なら k=0 にする
        ks = []
        valid_flags = []

        for f in features:
            text_ids = f["text_input_ids"]
            n_masks = self._count_masks(text_ids)

            valid = True
            if n_masks < 2 or (n_masks % 2) != 0:
                valid = False

            k = n_masks // 2 if valid else 0

            # ngram_token_start/end があるなら k と一致しているか
            nts = f.get("ngram_token_start", None)
            nte = f.get("ngram_token_end", None)
            if valid:
                if nts is not None and (not isinstance(nts, list) or len(nts) != k):
                    valid = False
                    k = 0
                if nte is not None and (not isinstance(nte, list) or len(nte) != k):
                    valid = False
                    k = 0

            # labels が list の場合は長さ k と一致しているべき（継続事前学習は [] OK）
            lbl = f.get("labels", None)
            if valid and lbl is not None and isinstance(lbl, list) and len(lbl) not in (0, k):
                valid = False
                k = 0

            ks.append(k)
            valid_flags.append(valid)

        # max_k は “有効サンプルの中の最大” で決める（全部無効でも 1 にしておく）
        max_k = max(ks) if len(ks) > 0 else 1
        if max_k == 0:
            max_k = 1

        # 2) tensor化 & pad（doc/text は全サンプル分）
        doc_input_ids = [torch.tensor(f["doc_input_ids"]).clone().detach() for f in features]
        doc_attention_mask = [torch.tensor(f["doc_attention_mask"]).clone().detach() for f in features]
        text_input_ids = [torch.tensor(f["text_input_ids"]).clone().detach() for f in features]
        text_attention_mask = [torch.tensor(f["text_attention_mask"]).clone().detach() for f in features]

        batch_doc_input_ids = pad_sequence(doc_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        batch_doc_attention_mask = pad_sequence(doc_attention_mask, batch_first=True, padding_value=0)
        batch_text_input_ids = pad_sequence(text_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        batch_text_attention_mask = pad_sequence(text_attention_mask, batch_first=True, padding_value=0)

        # 3) ngram_token_start/end は list のまま保持。
        #    無効サンプルは空にしておく（モデル側で valid_k=0 なので参照されない）
        ngram_token_start = []
        ngram_token_end = []
        for f, valid in zip(features, valid_flags):
            if valid:
                ngram_token_start.append(f.get("ngram_token_start", []))
                ngram_token_end.append(f.get("ngram_token_end", []))
            else:
                ngram_token_start.append([])
                ngram_token_end.append([])

        # 4) labels を (B, max_k) に pad
        labels_padded = []
        for f, k, valid in zip(features, ks, valid_flags):
            lbl = f.get("labels", None)

            if (not valid) or (k == 0):
                # 無効：全部 -100
                row = [self.pad_labels_value] * max_k
            else:
                if lbl is None or lbl == []:
                    # ラベル無し（継続事前学習）：全部 -100
                    row = [self.pad_labels_value] * max_k
                else:
                    # k 個分 + pad
                    row = list(lbl)[:k] + [self.pad_labels_value] * (max_k - k)

            labels_padded.append(row)

        labels = torch.tensor(labels_padded, dtype=torch.long)

        out = {
            "input_ids": [batch_doc_input_ids, batch_text_input_ids],
            "attention_mask": [batch_doc_attention_mask, batch_text_attention_mask],
            "ngram_token_start": ngram_token_start,   # B×k×m_l (list), invalidは []
            "ngram_token_end": ngram_token_end,       # B×k×m_l (list), invalidは []
            "labels": labels,                         # (B, max_k) pad済み
            "num_spans": torch.tensor(ks, dtype=torch.long),  # (B,)
            "is_valid": torch.tensor(valid_flags, dtype=torch.bool),  # デバッグ用（任意）
        }
        return out
######

def create_raw_dataset_inference(data):
    dataset = []
    for d in data:
        dataset.append(
            {
                "document": d["source_info"],
                "text": d["masked_text"],  # masked
                # "masked_span": d["masked_span"],
                "original_text": d["original_text"],
                "masked_span_index": d["masked_span_index"],
                "sentence_index": d["sentence_index"],
            }
        )
    dataset = Dataset.from_pandas(pd.DataFrame(dataset))
    return dataset

def tokenize_function_inference(examples, tokenizer):
    # examples はバッチ（list）を想定
    docs = examples["document"]
    texts = examples["text"]
    # masked_spans = examples["masked_span"]
    original_texts = examples["original_text"]
    masked_span_indexs = examples["masked_span_index"]
    starts = examples.get("ngram_start", [None] * len(docs))
    ends = examples.get("ngram_end", [None] * len(docs))
    sentence_ids = examples.get("sentence_index", [None] * len(docs))
    labels = examples.get("labels", [None] * len(docs))

    # ドキュメント側を fast tokenizer でトークナイズ（offset_mapping 必須）
    doc_enc = tokenizer(
        docs,
        truncation=True,
        max_length=8192,
        return_offsets_mapping=True,
        padding=False,
    )

    ngram_token_starts = []
    ngram_token_ends = []
    for i, offsets in enumerate(doc_enc["offset_mapping"]):
        start_char = starts[i]
        end_char = ends[i]
        if start_char is None or end_char is None:
            ngram_token_starts.append(None)
            ngram_token_ends.append(None)
            continue

        ts = None
        te = None
        # offsets は (start, end) のリスト（special token は (0,0)）
        for j, (o0, o1) in enumerate(offsets):
            if o0 == 0 and o1 == 0:
                # special token / padding, skip
                continue
            if ts is None and o0 <= start_char < o1:
                ts = j
            # end_char は exclusive、最初に end を超えるトークンを選ぶ
            if te is None and o1 >= end_char:
                te = j
                break

        # フォールバック：オーバーラップする最初／最後のトークンを探す
        if ts is None:
            for j, (o0, o1) in enumerate(offsets):
                if o0 < end_char and o1 > start_char:
                    ts = j
                    break
        if te is None:
            for j in range(len(offsets) - 1, -1, -1):
                o0, o1 = offsets[j]
                if o0 < end_char and o1 > start_char:
                    te = j
                    break

        ngram_token_starts.append(ts)
        ngram_token_ends.append(te)

    # offset_mapping はもう不要なので削除
    doc_enc.pop("offset_mapping", None)

    # text 側もトークナイズ（別キーで保持）
    text_enc = tokenizer(
        texts,
        truncation=True,
        max_length=8192,
        padding=False,
    )

    # masked_span_enc = tokenizer(masked_spans,truncation=True,max_length=128,padding=False,)
    original_enc = tokenizer(
        original_texts,
        truncation=True,
        max_length=8192,
        padding=False,
        return_offsets_mapping=True,
    )
    masked_span_token_indices = []
    for i, offsets in enumerate(original_enc["offset_mapping"]):
        start_char, end_char = masked_span_indexs[i]
        if start_char is None or end_char is None:
            masked_span_token_indices.append((None, None))
            continue

        ts = None
        te = None
        # offsets は (start, end) のリスト（special token は (0,0)）
        for j, (o0, o1) in enumerate(offsets):
            if o0 == 0 and o1 == 0:
                continue
            if ts is None and o0 <= start_char < o1:
                ts = j
            if te is None and o1 >= end_char:
                te = j
                break

        # フォールバック：オーバーラップする最初／最後のトークンを探す
        if ts is None:
            for j, (o0, o1) in enumerate(offsets):
                if o0 < end_char and o1 > start_char:
                    ts = j
                    break
        if te is None:
            for j in range(len(offsets) - 1, -1, -1):
                o0, o1 = offsets[j]
                if o0 < end_char and o1 > start_char:
                    te = j
                    break

        # 範囲チェック
        if ts is None or te is None or te < ts:
            masked_span_token_indices.append((None, None))
        else:
            masked_span_token_indices.append((ts, te + 1))  # python slice で end は exclusive

    # offset_mapping は不要なので削除
    original_enc.pop("offset_mapping", None)

    # 戻り値：トークナイズした document / text 両方と ngram_token_start/end, labels
    out = {
        "doc_input_ids": doc_enc["input_ids"],
        "doc_attention_mask": doc_enc["attention_mask"],
        "text_input_ids": text_enc["input_ids"],
        "text_attention_mask": text_enc["attention_mask"],
        # "masked_span_input_ids": masked_span_enc["input_ids"],
        # "masked_span_attention_mask": masked_span_enc["attention_mask"],
        "original_input_ids": original_enc["input_ids"],
        "original_attention_mask": original_enc["attention_mask"],
        "masked_span_index": masked_span_token_indices,
        "ngram_token_start": ngram_token_starts,
        "ngram_token_end": ngram_token_ends,
        "sentence_index": sentence_ids,
        "labels": labels,
    }
    return out

def create_datasets_inference(data, tokenizer):
    dataset = create_raw_dataset_inference(data)
    raw_dataset = DatasetDict(
        {
            "test": dataset,
        }
    )
    tokenized_dataset = raw_dataset.map(
        lambda x: tokenize_function_inference(x, tokenizer),
        batched=True,
        remove_columns=raw_dataset["test"].column_names,
    )
    return tokenized_dataset

######