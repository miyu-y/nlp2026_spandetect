import re
from tqdm import tqdm

def include_ngram(
    text: str,
    doc: str,
    n: int | None = None,
    min_n: int = 1,
    max_n: int | None = None,
    tokenizer=None,
    lower: bool = True,
):
    """
    text の n-gram が doc に含まれるかを返す。
    - n を指定するとその長さのみをチェック。
    - n を None にすると min_n..max_n の長さをチェック（max_n default = len(text_tokens)）。
    - tokenizer が与えられれば tokenizer.tokenize(...).text を使う。
    戻り値:
        (included: bool, matches: list[str] | None)
        matches は doc 側で見つかった n-gram の表層（重複なし）
    """
    def tok_to_list(s):
        s = s.replace(".", "").replace(",", "").replace(":", "").replace(";", "")
        if tokenizer is None:
            toks = s.split()
        else:
            toks = [t.text for t in tokenizer.tokenize(s)]
        return [t.lower() for t in toks] if lower else toks

    t_toks = tok_to_list(text)
    d_toks = tok_to_list(doc)
    if not t_toks or not d_toks:
        return False, None

    L = len(t_toks)
    max_n = (max_n or L)
    min_n = max((L + 2) // 3, min_n)

    if n is not None:
        lengths = [n]
    else:
        lengths = range(max(min_n, 1), min(max_n, L) + 1)

    # 長い n から優先して探し、最初に見つかった長さ l について
    # その l の全マッチを集めて返す or 許容される長さのやつは全部
    all_found_phrases = []
    for l in sorted(lengths, reverse=True):
        if l > len(d_toks) or l > len(t_toks):
            continue

        # text 側の全 n-gram を集合に
        t_ngrams = {tuple(t_toks[i:i+l]) for i in range(len(t_toks) - l + 1)}

        # 前置詞、冠詞、代名詞を除去（l==1 のときだけ）
        if max_n > 1 and l == 1:
            stopwords = {
                "the", "a", "an", "in", "on", "at", "for", "to", "of", "by", "with",
                "and", "from", "as", "before", "after", "about", "into", "over",
                "under", "between", "among", "near", "until", "is", "are", "was",
                "were", "he", "she", "it", "they", "we", "you", "i", "me", "him",
                "her", "them", "us", "my", "your", "his", "its", "our", "their",
                "this", "that", "these", "those", "it"
            }
            t_ngrams = {ng for ng in t_ngrams if ng[0] not in stopwords}

        if not t_ngrams:
            continue

        # doc をスライドして一致を探し、同じ長さ l のマッチを全部集める
        found_phrases = []
        for j in range(len(d_toks) - l + 1):
            span = tuple(d_toks[j:j+l])
            if span in t_ngrams:
                found_phrases.append(" ".join(d_toks[j:j+l]))

        if found_phrases:
            # 重複を消して返す（順序はとりあえず保持）
            seen = set()
            uniq = []
            for ph in found_phrases:
                if ph not in seen:
                    seen.add(ph)
                    uniq.append(ph)
            #return True, uniq
            all_found_phrases.extend(found_phrases)
    if all_found_phrases:
        return True, all_found_phrases

    return False, None

def include_ngram_perfect(text: str, doc: str) -> bool:
    """ text の完全一致が doc に含まれるかを返す。
    大文字小文字は区別しない。
    """
    pattern = re.escape(text)
    match = re.search(pattern, doc, flags=re.IGNORECASE)
    if match:
        # マッチした文字列の全てのリストを返す
        return True, [match.group(0)]
    else:
        return False, None

def search_ngram(data, perfect: bool = False):
    for d in tqdm(data):
        source_info = d["source_info"]
        source_info = source_info["passages"] if isinstance(source_info, dict) else source_info
        #response = d["response"]

        hal_chars = set()
        for l in d["labels"]:
            hal_chars.update(range(l["start"], l["end"]))

        for s in d["srl_splits"]:
            s["hallucinated"] = len(set(range(s["start"], s["end"])) & hal_chars) > 0
            if perfect:
                included, matches = include_ngram_perfect(s["text"], source_info)
            else:   
                included, matches = include_ngram(s["text"], source_info)
            s["included"] = included

            # ここを複数対応に変更
            if not included or matches is None:
                s["ngram_match"] = []
                s["ngram_start"] = []
                s["ngram_end"] = []
            else:
                # matches: ["phrase1", "phrase2", ...]
                start_list = []
                end_list = []
                for phrase in matches:
                    # phrase が doc 内に出現する全ての位置を取得
                    pattern = re.escape(phrase)
                    for m in re.finditer(pattern, source_info, flags=re.IGNORECASE):
                        start_list.append(m.start())
                        end_list.append(m.end())

                s["ngram_match"] = matches            # フレーズのリスト
                s["ngram_start"] = start_list         # char start のリスト
                s["ngram_end"] = end_list             # char end のリスト

        d["source_info"] = source_info
        d["hal_chars"] = hal_chars
    return data

