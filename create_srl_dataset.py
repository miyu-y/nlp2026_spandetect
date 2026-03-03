from allennlp.predictors.predictor import Predictor
from allennlp.data.tokenizers import SpacyTokenizer
import json
import re
import argparse
from tqdm import tqdm

from transformers import AutoTokenizer
import torch

parser = argparse.ArgumentParser(description="SRL split script")
parser.add_argument("--mode", choices=["train", "dev", "test"], default="test",
                    help="Which split to process (default: test)")
args = parser.parse_args()
mode = args.mode
print(f"Mode: {mode}")

_SENT_END = re.compile(r'([.!?])([)"\]]*)\s+')
_PUNCT_ONLY = re.compile(r'^[\s\.,:;!\?\-\(\)\[\]\"\'`]+$')  # 句読点だけ


def _split_sentences_with_offsets(text: str):
    """
    text を文ごとに {"text","start","end"} のリストへ。
    末尾に終止記号が無い最終文も拾う。
    """
    sents = []
    start = 0
    for m in _SENT_END.finditer(text):
        end = m.end(1) + (len(m.group(2)) if m.group(2) else 0)
        sents.append({"text": text[start:end], "start": start, "end": end})
        start = m.end()
    if start < len(text):
        sents.append({"text": text[start:], "start": start, "end": len(text)})
    return sents


def _char_spans_from_words(words, sent_text):
    """
    SRL の words を元の sent_text 上の char span にざっくりアライン。
    """
    spans = []
    i = 0
    for w in words:
        while i < len(sent_text) and sent_text[i].isspace():
            i += 1
        cands = [w, w.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')]
        j = -1
        use = w
        for c in cands:
            j = sent_text.find(c, i)
            if j >= 0:
                use = c
                break
        if j < 0:
            spans.append((i, i))
        else:
            spans.append((j, j + len(use)))
            i = j + len(use)
    return spans


def _collect_boundaries_from_verbs(verbs_tags):
    """
    複数の述語タグ列（BIO付き）から、
    「どこでラベルが切り替わるか」を集約して最小分割の境界集合を作る。
    """
    if not verbs_tags:
        return set()

    N = len(verbs_tags[0])
    boundaries = {0, N}

    for tags in verbs_tags:
        in_seg = False
        cur_label = None
        for i, tag in enumerate(tags):
            if tag == "O":
                if in_seg:
                    boundaries.add(i)
                    in_seg = False
                    cur_label = None
            elif tag.startswith("B-"):
                if in_seg:
                    boundaries.add(i)
                boundaries.add(i)
                in_seg = True
                cur_label = tag[2:]
            elif tag.startswith("I-"):
                lab = tag[2:]
                if not in_seg:
                    boundaries.add(i)
                    in_seg = True
                    cur_label = lab
                elif lab != cur_label:
                    boundaries.add(i)
                    cur_label = lab
        if in_seg:
            boundaries.add(N)

    return set(sorted(boundaries))


def _labels_in_span_with_index(verbs, i, j):
    """
    verbs: List[{"tags": [...]}]  # allennlp の出力
    [i, j) のトークン範囲に重なる非 O タグを
      - labels: ["ARG1","V", ...]
      - labels_indexed: ["0-ARG1","2-V", ...]  # verbs のインデックス付き
    で返す（B-/I- は正規化して落とす）。
    """
    labels = set()
    labels_indexed = set()
    for vi, v in enumerate(verbs):
        tags = v.get("tags")
        if not tags:
            continue
        for t in tags[i:j]:
            if t == "O":
                continue
            lab = t[2:] if "-" in t else t  # "B-ARG1" / "I-ARG1" / "V" → "ARG1" / "V"
            labels.add(lab)
            labels_indexed.add(f"{vi}-{lab}")
    return sorted(labels), sorted(labels_indexed)


def split_spans_srl(text: str):
    """
    SRL の全述語解析結果を統合して「一番細かい単位」で分割するだけの関数。

    - 全ての述語のタグ列から境界を集めて最小分割を作る
    - O しか無い区間は削除
    - 句読点だけのスパンも削除
    - 各スパンについて:
        - text, start, end (char offset)
        - labels: ["ARG0","V", ...]
        - labels_indexed: ["0-ARG0","1-V", ...]
        - sentence_index, token_span を保持
    """
    results = []
    sentences = _split_sentences_with_offsets(text)

    for s_idx, sent in enumerate(sentences):
        sent_text = sent["text"]
        base = sent["start"]

        out = srl.predict(sentence=sent_text)
        verbs = out.get("verbs", None)
        if not verbs:
            continue

        words = out["words"]
        offsets_local = _char_spans_from_words(words, sent_text)

        # 全ての述語で token 数が一致するものだけ使う
        verbs_tags = [v["tags"] for v in verbs if "tags" in v and len(v["tags"]) == len(words)]
        if not verbs_tags:
            continue

        boundaries = sorted(_collect_boundaries_from_verbs(verbs_tags))

        for left, right in zip(boundaries[:-1], boundaries[1:]):
            if left == right:
                continue

            # token span → char span
            c_start = offsets_local[left][0] if left < len(offsets_local) else offsets_local[-1][1]
            c_end   = offsets_local[right - 1][1] if right - 1 < len(offsets_local) else offsets_local[-1][1]
            if c_end < c_start:
                c_end = c_start

            labs, labs_idx = _labels_in_span_with_index(verbs, left, right)
            if not labs:  # 全解析で O → 除外
                continue

            span_text = sent_text[c_start:c_end]
            if _PUNCT_ONLY.match(span_text):  # 句読点のみ → 除外
                continue

            results.append({
                "text": span_text,
                "start": base + c_start,
                "end":   base + c_end,
                "labels": labs,
                "labels_indexed": labs_idx,
                "sentence_index": s_idx,
                "token_span": (left, right),
            })

    # 文順・開始位置でソートして返す
    results.sort(key=lambda x: (x["sentence_index"], x["start"]))
    if results!=[]:
        results = merge_verbs(results,text)
        results = merge_single_word(results,text)
    return results

def merge_verbs(span_list,text):
    """
    - V以外のラベルは同じ
    - V同士で隣り合っている (同じ文で、句読点とかで区切られてない)
    スパンを統合する。
    """
    new_list = [span_list[0]]
    span_list = span_list[1:]
    while span_list:
        span = new_list[-1]
        next_span = span_list[0]
        if "V" in span["labels"]:
            labels_wo_v = [l for l in span["labels_indexed"] if not l.endswith("-V")]
            next_labels_wo_v = [l for l in next_span["labels_indexed"] if not l.endswith("-V")]
            if "V" in next_span["labels"] and labels_wo_v==next_labels_wo_v and span["sentence_index"] == next_span["sentence_index"] and span["end"]+1==next_span["start"]:
                new_list = new_list[:-1]
                # 統合
                merged_span = {
                    "text": text[span["start"]:next_span["end"]],
                    "start": span["start"],
                    "end": next_span["end"],
                    "labels": ["V"],
                    "labels_indexed": [span["labels_indexed"][0], next_span["labels_indexed"][0]],
                    "sentence_index": span["sentence_index"],
                    "token_span": (span["token_span"][0], next_span["token_span"][1]),
                }
                new_list.append(merged_span)
            else:
                new_list.append(next_span)
        else:
            new_list.append(next_span)
        span_list = span_list[1:]
    return new_list

def merge_single_word(span_list, text):
    """
    - 1wordだけのスパン & ラベル集合が隣のスパンの部分集合
    スパンを統合する。
    """
    new_list = [span_list[0]]
    span_list = span_list[1:]
    while span_list:
        span = new_list[-1]
        next_span = span_list[0]
        prev_span = new_list[-2] if len(new_list) > 1 else None
        if len(span["text"].split())==1:
            if set(span["labels_indexed"]).issubset(set(next_span["labels_indexed"])) and span["sentence_index"] == next_span["sentence_index"]:
                new_list = new_list[:-1]
                # 統合
                merged_span = {
                    "text": text[span["start"]:next_span["end"]],
                    "start": span["start"],
                    "end": next_span["end"],
                    "labels": next_span["labels"],
                    "labels_indexed": next_span["labels_indexed"],
                    "sentence_index": span["sentence_index"],
                    "token_span": (span["token_span"][0], next_span["token_span"][1]),
                }
                new_list.append(merged_span)
            elif prev_span and set(span["labels_indexed"]).issubset(set(prev_span["labels_indexed"])) and span["sentence_index"] == prev_span["sentence_index"]:
                new_list = new_list[:-2]
                # 統合
                merged_span = {
                    "text": text[prev_span["start"]:span["end"]],
                    "start": prev_span["start"],
                    "end": span["end"],
                    "labels": prev_span["labels"],
                    "labels_indexed": prev_span["labels_indexed"],
                    "sentence_index": span["sentence_index"],
                    "token_span": (prev_span["token_span"][0], span["token_span"][1]),
                }
                new_list.append(merged_span)
            else:
                new_list.append(next_span)
        else:
            new_list.append(next_span)
        span_list = span_list[1:]
    return new_list

def build_sentence_ids_with_regex(text: str, doc_input_ids):
    """
    text          : 1つの doc の元テキスト（source / response 等）
    tokenizer     : facebook/npm の tokenizer
    doc_input_ids : そのテキストを tokenizer したときの input_ids
                    （add_special_tokens=True を想定）

    戻り値:
        sentence_ids: len(doc_input_ids) と同じ長さのリスト
                      special token には -1、それ以外は文ID(0,1,2,...)。
    """
    # doc_input_ids を list にそろえる
    if isinstance(doc_input_ids, torch.Tensor):
        doc_ids = doc_input_ids.cpu().tolist()
    else:
        doc_ids = list(doc_input_ids)

    # 1. 文ごとに分割
    sent_infos = _split_sentences_with_offsets(text)  # [{"text","start","end"}, ...]

    # 2. 各文を add_special_tokens=False でトークナイズして token 列を取得
    sent_token_ids = []
    for s in sent_infos:
        enc = tokenizer(
            s["text"],
            add_special_tokens=False,
        )
        sent_token_ids.append(enc["input_ids"])

    # 3. special token ID のセットを作る
    pad_id = tokenizer.pad_token_id
    cls_id = getattr(tokenizer, "cls_token_id", None)
    sep_id = getattr(tokenizer, "sep_token_id", None)
    bos_id = getattr(tokenizer, "bos_token_id", None)
    eos_id = getattr(tokenizer, "eos_token_id", None)

    special_ids = {pad_id}
    for x in [cls_id, sep_id, bos_id, eos_id]:
        if x is not None:
            special_ids.add(x)

    # 4. doc_ids を左から走査しながら、special 以外に文IDをふっていく
    sentence_ids = []
    cur_sent_idx = 0
    pos_in_sent = 0  # 現在の文の中での token 位置

    for tid in doc_ids:
        if tid in special_ids:
            # special token は span 探索に使わないので -1
            sentence_ids.append(-1)
            continue

        # plain token の割り当てが終わっている場合
        if cur_sent_idx >= len(sent_token_ids):
            sentence_ids.append(-1)
            continue

        cur_sent_tokens = sent_token_ids[cur_sent_idx]
        sentence_ids.append(cur_sent_idx)

        pos_in_sent += 1
        if pos_in_sent >= len(cur_sent_tokens):
            # この文の token を使い切ったら次の文へ
            cur_sent_idx += 1
            pos_in_sent = 0

    if len(sentence_ids) != len(doc_ids):
        print(
            f"[WARN] sentence_ids length {len(sentence_ids)} != doc_ids length {len(doc_ids)}"
        )

    return sentence_ids

# 1) SRL予測器のロード（既定の英語SRL）
srl = Predictor.from_path(  # 既定のSRLモデルを取得
    "https://storage.googleapis.com/allennlp-public-models/bert-base-srl-2020.11.19.tar.gz"
)
tok = SpacyTokenizer(language="en_core_web_sm")  # 文字オフセット取得用
tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-large")

with open("data/ft_{}.jsonl".format(mode), "r") as f:
    data = [json.loads(line) for line in f]
    allowed_keys = ["id_name", "source_id", "task_type", "model", "source",
                "source_info", "response", "labels", "split"]
    data = [{k: v for k, v in d.items() if k in allowed_keys} for d in data]
        
cover_rate = []
full_size = []
part_size = []
split_list = []
for d in tqdm(data):
    span_list = split_spans_srl(d["response"])
    d["srl_splits"] = span_list
    document = d["source_info"] if type(d.get("source_info"))==str else d["source_info"]["passages"]
    doc_input_ids = tokenizer.encode(document, add_special_tokens=True, padding = True)
    sent_ids = build_sentence_ids_with_regex(document, doc_input_ids)
    d["sentence_ids"] = sent_ids
    split_list.append(span_list)
    full_hallucinated = set()
    for l in d["labels"]:
        full_hallucinated.update(range(l["start"], l["end"]))
    part_hallucinated = set()
    for sp in span_list:
        part_hallucinated.update(range(sp["start"], sp["end"]))
    rate = len(full_hallucinated & part_hallucinated) / len(full_hallucinated) if len(full_hallucinated) > 0 else 0.0
    cover_rate.append(rate)
    full_size.append(len(full_hallucinated))
    part_size.append(len(part_hallucinated))

print(f"Average coverage rate (macro): {sum(cover_rate)/len(cover_rate):.3f}")
print(f"Average coverage rate (micro): {sum(cover_rate[i]*full_size[i] for i in range(len(cover_rate)))/sum(full_size):.3f}")
print(f"Average full hallucinated size: {sum(full_size)/len(full_size):.1f}")
print(f"Average part hallucinated size: {sum(part_size)/len(part_size):.1f}")

with open(f"data/1127_srl_{mode}.jsonl", "w") as f:
    for d in data:
        f.write(json.dumps(d, ensure_ascii=False) + "\n")
hal_data = [d for d in data if d["labels"]!=[]]
with open(f"data/1127_srl_{mode}_hal.jsonl", "w") as f:
    for d in hal_data:
        f.write(json.dumps(d, ensure_ascii=False) + "\n")