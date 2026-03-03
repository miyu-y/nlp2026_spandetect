import re, json
from typing import List, Tuple, Set

with open("data/ft_test.jsonl") as f:
    test_data = [json.loads(line) for line in f]
output_file_path = "ft_inference_results/1027_preds_lora.jsonl"


def extract_hallucination_list(raw: str) -> List[str]:
    """表記ゆれを許容して 'hallucination list' のリスト中身を抜き出す。
    返り値は文字列リスト（要素はトリム済み）。見つからなければ []。
    """
    if not raw:
        return []
    # 正規の JSON 部分をまず試す
    try:
        # raw に余分な前後があれば JSON 部分だけ切り出す
        jstart = raw.find("{")
        jend = raw.rfind("}")
        if jstart >= 0 and jend > jstart:
            sub = raw[jstart : jend + 1]
            obj = json.loads(sub)
            if isinstance(obj, dict) and "hallucination list" in obj:
                v = obj["hallucination list"]
                return [str(x).strip() for x in v] if isinstance(v, list) else []
    except Exception:
        pass

    # regex で [] 内の内容を探す（表記ゆれに寛容）
    pat_list = [
        r"\"hallucination\s*list\"\s*:\s*\[(.*?)\]",  # "hallucination list": [...]
        r"hallucinat(?:e|ion)\s*list\s*[:=]\s*\[(.*?)\]",  # hallucination list: [...]
        r"\{.*?hallucinat(?:e|ion).*?\[(.*?)\].*?\}",  # {...halluc... [ ... ] ...}
    ]
    inner = None
    for p in pat_list:
        m = re.search(p, raw, flags=re.IGNORECASE | re.DOTALL)
        if m:
            inner = m.group(1)
            break
    if inner is None:
        # 最後の砦：全角/半角の [] を探す
        m = re.search(r"\[([^\]]*?)\]", raw, flags=re.DOTALL)
        inner = m.group(1) if m else None

    if not inner:
        # print("Warning: could not extract hallucination list from raw:", raw)
        return []

    # inner の中から "..." や '...' や カンマ区切りの裸文字列を列挙
    spans = []
    for qm in re.finditer(r'"([^"]{1,2000}?)"|\'([^\']{1,2000}?)\'|([^,\n\r]+)', inner):
        g = qm.group(1) or qm.group(2) or qm.group(3)
        if g is None:
            continue
        # カンマや末尾のスペースを取り除く
        g = g.strip().strip(",").strip()
        if not g:
            continue
        spans.append(g)
    # さらに不要なゴミ行（例 "hallucination list": [] の残骸）を除去
    spans = [s for s in spans if "hallucination" not in s.lower()][:200]
    return spans


def normalize_with_map(s: str) -> Tuple[str, List[int]]:
    """空白を正規化（連続空白→1つ）、戻せる位置マップを返す。
    normalized[i] corresponds to original index map[i].
    """
    out = []
    map_idx = []
    prev_space = False
    for i, ch in enumerate(s):
        if ch.isspace():
            if not prev_space:
                out.append(" ")
                map_idx.append(i)
            prev_space = True
        else:
            out.append(ch)
            map_idx.append(i)
            prev_space = False
    norm = "".join(out).strip()
    # adjust mapping if we stripped leading spaces
    if norm and map_idx:
        # if we stripped leading space, mapping already points to first original non-space
        pass
    return norm, map_idx


def find_span_positions(text: str, span_text: str) -> List[Tuple[int, int]]:
    """span_text を text 中で見つけて (start,end) のリストを返す（end は exclusive）。
    まず厳密検索、ダメなら正規化して検索。それでもダメなら空リスト。
    """
    if not span_text:
        return []
    # 1) 直接検索
    idx = text.find(span_text)
    if idx >= 0:
        return [(idx, idx + len(span_text))]
    # 2) 小さな変換（先頭末尾の空白・句読点除去）
    s2 = span_text.strip(" \t\n\r.,;:—-")
    if s2 and s2 != span_text:
        idx = text.find(s2)
        if idx >= 0:
            return [(idx, idx + len(s2))]
    # 3) 正規化して検索（空白連続を単一にする）
    norm_text, map_text = normalize_with_map(text)
    norm_span, map_span = normalize_with_map(span_text)
    if not norm_span:
        return []
    pos = norm_text.find(norm_span)
    if pos >= 0:
        # map_text[pos] は norm_text の pos の元の index
        orig_start = map_text[pos]
        # end: map_text[pos + len(norm_span)-1] +1
        last_norm_pos = pos + len(norm_span) - 1
        orig_end = map_text[last_norm_pos] + 1
        return [(orig_start, orig_end)]
    # 4) 部分一致で最長一致を探す（最初の 20 文字の断片など）
    frag = norm_span[: min(40, len(norm_span))]
    pos = norm_text.find(frag)
    if pos >= 0:
        orig_start = map_text[pos]
        # try expand until we approximate span length
        orig_end = orig_start + len(span_text)
        orig_end = min(len(text), orig_end)
        return [(orig_start, orig_end)]
    return []


def predicted_char_set_from_spans(response_text: str, pred_spans: List[str]) -> Set[int]:
    """pred_spans（文字列リスト）を response_text 上にマップして文字 idx の集合を返す"""
    s = set()
    for ps in pred_spans:
        pos_list = find_span_positions(response_text, ps)
        for a, b in pos_list:
            s.update(range(a, b))
    return s


def gold_char_set_from_labels(item: dict) -> Set[int]:
    """item の labels から gold の文字 idx セットを返す"""
    s = set()
    for lab in item.get("labels", []):
        a = int(lab.get("start", 0))
        b = int(lab.get("end", 0))
        if b > a:
            s.update(range(a, b))
    return s


def char_metrics_from_sets(gold: Set[int], pred: Set[int]) -> dict:
    tp = len(gold & pred)
    fp = len(pred - gold)
    fn = len(gold - pred)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    acc = len(gold & pred) + len(
        (set(range(max(max(gold, default=-1), max(pred, default=-1) + 1) + 1)) - (gold | pred))
    )
    # accuracy at character-level is ambiguous if document lengths vary; return basic stats
    return {"TP": tp, "FP": fp, "FN": fn, "Precision": prec, "Recall": rec, "F1": f1}


def evaluate_char_level(test_data: List[dict], preds_by_id: dict) -> dict:
    """test_data の各 item と preds_by_id[item_id]（raw 文字列）を比較して集計を返す。
    preds_by_id のキーは item.get('id') または item.get('id_name') 等に合わせて用意すること。
    """
    total_tp = total_fp = total_fn = 0
    per_item = []
    for item in test_data:
        # id 判定（柔軟）
        idk = item.get("id") or item.get("id_name") or str(item.get("source_id", ""))
        raw = preds_by_id.get(idk)
        if raw is None:
            # try numeric id keys
            raw = preds_by_id.get(str(idk))
        pred_spans = extract_hallucination_list(raw if raw is not None else "")
        resp = item.get("response", "")
        pred_set = predicted_char_set_from_spans(resp, pred_spans)
        gold_set = gold_char_set_from_labels(item)
        tp = len(gold_set & pred_set)
        fp = len(pred_set - gold_set)
        fn = len(gold_set - pred_set)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        per_item.append({"id": idk, "TP": tp, "FP": fp, "FN": fn, "pred_spans": pred_spans})
    prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {
        "TP": total_tp,
        "FP": total_fp,
        "FN": total_fn,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "per_item": per_item,
    }


def eval_type_recall(test_data: List[dict], preds_by_id: dict, type_list: List[str]) -> dict:
    """test_data の各 item と preds_by_id[item_id]（raw 文字列）を比較してタイプ別リコールを返す。
    preds_by_id のキーは item.get('id') または item.get('id_name') 等に合わせて用意すること。
    type_list は評価対象のタイプ名リスト。
    """
    type_to_total = {t: 0 for t in type_list}
    type_to_found = {t: 0 for t in type_list}
    for item in test_data:
        # id 判定（柔軟）
        idk = item.get("id") or item.get("id_name") or str(item.get("source_id", ""))
        raw = preds_by_id.get(idk)
        if raw is None:
            # try numeric id keys
            raw = preds_by_id.get(str(idk))
        pred_spans = extract_hallucination_list(raw if raw is not None else "")
        resp = item.get("response", "")
        pred_set = predicted_char_set_from_spans(resp, pred_spans)
        for lab in item.get("labels", []):
            typ = lab.get("label_type", "").strip()
            if typ not in type_list:
                continue
            a = int(lab.get("start", 0))
            b = int(lab.get("end", 0))
            if b > a:
                type_to_total[typ] += 1
                if any(idx in pred_set for idx in range(a, b)):
                    type_to_found[typ] += 1
    type_to_recall = {}
    for t in type_list:
        total = type_to_total[t]
        found = type_to_found[t]
        rec = found / total if total > 0 else 0.0
        type_to_recall[t] = {"Total": total, "Found": found, "Recall": rec}
    return type_to_recall


with open(output_file_path, "r") as f:
    preds_by_id = {d["id"]: d["raw"] for d in [json.loads(line) for line in f]}
# hallucinatedの例だけで
test_data_hal = [item for item in test_data if item.get("labels")]
test_data_hal_qa = [item for item in test_data if item.get("labels") and item.get("task_type") == "QA"]
test_data_hal_sum = [item for item in test_data if item.get("labels") and item.get("task_type") == "Summary"]
metrics_hal = evaluate_char_level(test_data_hal, preds_by_id)
metrics_hal_qa = evaluate_char_level(test_data_hal_qa, preds_by_id)
metrics_hal_sum = evaluate_char_level(test_data_hal_sum, preds_by_id)
print("Hallucinated only:", metrics_hal["Precision"], metrics_hal["Recall"], metrics_hal["F1"])
print("Hallucinated QA:", metrics_hal_qa["Precision"], metrics_hal_qa["Recall"], metrics_hal_qa["F1"])
print("Hallucinated Summary:", metrics_hal_sum["Precision"], metrics_hal_sum["Recall"], metrics_hal_sum["F1"])

metric_conflict = eval_type_recall(test_data_hal, preds_by_id, ["Evident Conflict", "Subtle Conflict"])
metric_baseless = eval_type_recall(test_data_hal, preds_by_id, ["Evident Baseless Info", "Subtle Baseless Info"])
print("Conflict type recall:", metric_conflict)
print("Baseless type recall:", metric_baseless)
