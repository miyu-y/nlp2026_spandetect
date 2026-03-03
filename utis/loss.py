# utis/loss.py
import math
import torch
import torch.nn.functional as F

# ======================
# 共通ヘルパ
# ======================

def _prepare_sims_for_one(
    q_s_row,
    q_e_row,
    y_s_plus_i,
    y_s_minus_i,
    y_e_plus_i,
    y_e_minus_i,
    device,
    normalize: bool = True,
):
    """
    1サンプル分について
    - list→tensor
    - 空チェック
    - （必要なら）正規化
    - sim_s_p, sim_s_m, sim_e_p, sim_e_m, sim_s_all, sim_e_all を返す
    """
    H = q_s_row.size(-1)

    def _to_tensor(x):
        if isinstance(x, list):
            x = torch.stack(x, dim=0)
        return x.to(device)

    ys_p = _to_tensor(y_s_plus_i)
    ys_m = _to_tensor(y_s_minus_i)
    ye_p = _to_tensor(y_e_plus_i)
    ye_m = _to_tensor(y_e_minus_i)

    # 空ならゼロ埋め（本当はスキップ推奨だけど、今の挙動は維持）
    if ys_p.numel() == 0:
        ys_p = torch.zeros(1, H, device=device)
    if ys_m.numel() == 0:
        ys_m = torch.zeros(1, H, device=device)
    if ye_p.numel() == 0:
        ye_p = torch.zeros(1, H, device=device)
    if ye_m.numel() == 0:
        ye_m = torch.zeros(1, H, device=device)

    q_s = q_s_row.unsqueeze(0)  # (1,H)
    q_e = q_e_row.unsqueeze(0)  # (1,H)

    if normalize:
        q_s = F.normalize(q_s, dim=-1)
        q_e = F.normalize(q_e, dim=-1)
        ys_p = F.normalize(ys_p, dim=-1)
        ys_m = F.normalize(ys_m, dim=-1)
        ye_p = F.normalize(ye_p, dim=-1)
        ye_m = F.normalize(ye_m, dim=-1)

    sim_s_p = (q_s @ ys_p.T)   # (1, Ns+)
    sim_s_m = (q_s @ ys_m.T)   # (1, Ns-)
    sim_e_p = (q_e @ ye_p.T)   # (1, Ne+)
    sim_e_m = (q_e @ ye_m.T)   # (1, Ne-)

    sim_s_all = torch.cat([sim_s_p, sim_s_m], dim=1)  # (1, Ns+ + Ns-)
    sim_e_all = torch.cat([sim_e_p, sim_e_m], dim=1)

    return sim_s_p, sim_s_m, sim_e_p, sim_e_m, sim_s_all, sim_e_all


# ======================
# 1) 学習用の loss 関数
# ======================

def compute_loss_train(
    mask_s_output,
    mask_e_output,
    labels,
    y_s_plus,
    y_s_minus,
    y_e_plus,
    y_e_minus,
    normalize: bool = True,
    mode: str = None,
    pad_label: int = -100,   # collatorで使ったpad値
):
    """
    学習時に Trainer から呼ばれる損失関数。
    multi-span 対応：
      - mask_s_output/mask_e_output が (N,H) でも (B,K,H) でもOK
      - labels が (N,) / (B,K) / None でもOK
      - labels==pad_label は loss から除外
    """
    device = mask_s_output.device

    # ---------
    # 0) (B,K,H) -> (N,H) に flatten（必要なら）
    # ---------
    if mask_s_output.dim() == 3:
        # (B,K,H)
        B, K, H = mask_s_output.size()
        mask_s_flat = mask_s_output.reshape(B * K, H)
        mask_e_flat = mask_e_output.reshape(B * K, H)
    else:
        # (N,H)
        mask_s_flat = mask_s_output
        mask_e_flat = mask_e_output
        H = mask_s_flat.size(1)

    N = mask_s_flat.size(0)

    # ---------
    # 1) labels を (N,) に整形し、pad を除外するための valid mask を作る
    # ---------
    if labels is None:
        labels_flat = None
        valid = None
    else:
        if isinstance(labels, list):
            labels = torch.tensor(labels, device=device)

        if labels.dim() == 2:
            # (B,K) -> (N,)
            labels_flat = labels.reshape(-1).to(device)
        else:
            # (N,)
            labels_flat = labels.to(device)

        if labels_flat.numel() != N:
            raise ValueError(f"labels length {labels_flat.numel()} != N_spans {N}")

        valid = (labels_flat != pad_label)

    # ---------
    # 2) spanごとに loss/score を計算
    # ---------
    losses = []
    scores = []

    for i in range(N):
        sim_s_p, sim_s_m, sim_e_p, sim_e_m, sim_s_all, sim_e_all = _prepare_sims_for_one(
            mask_s_flat[i],
            mask_e_flat[i],
            y_s_plus[i],
            y_s_minus[i],
            y_e_plus[i],
            y_e_minus[i],
            device=device,
            normalize=normalize,
        )

        # スコア（参考出力）
        # sim_*_all: (1, M)
        max_s = sim_s_all.max(dim=1).values  # (1,)
        max_e = sim_e_all.max(dim=1).values  # (1,)
        score_i = torch.exp(max_s) + torch.exp(max_e)  # (1,)
        scores.append(score_i.squeeze(0))  # scalar

        # loss（labels があり、かつ pad ではないものだけ）
        if labels_flat is not None:
            if not bool(valid[i].item()):
                continue
            h = labels_flat[i].float()
            
            if mode == "margin":
                loss_i = margin_loss(sim_s_p, sim_s_m, sim_e_p, sim_e_m, h,margin=0.2)  # (1,)
            else:
                loss_i = contrastive_loss(sim_s_p, sim_s_m,sim_e_p, sim_e_m,sim_s_all, sim_e_all,h)  # (1,)
            losses.append(loss_i.squeeze(0))

    score_tensor = torch.stack(scores)  # (N,)
    loss = torch.stack(losses).mean() if (labels_flat is not None and len(losses) > 0) else None
    #losses_tensor = torch.stack(losses)  # (n_valid,)
    #gamma = 2.0  # 1~3あたりで
    #w = (1.0 - torch.exp(-losses_tensor.detach())).pow(gamma)  # 0~1の重み
    #loss = (w * losses_tensor).sum() / (w.sum() + 1e-12)

    return loss, score_tensor

def contrastive_loss(sim_s_p, sim_s_m, sim_e_p, sim_e_m, sim_s_all, sim_e_all, h):
    log_pos_s = torch.logsumexp(sim_s_p, dim=1)
    log_neg_s = torch.logsumexp(sim_s_m, dim=1)
    log_pos_e = torch.logsumexp(sim_e_p, dim=1)
    log_neg_e = torch.logsumexp(sim_e_m, dim=1) 
    log_all_s = torch.logsumexp(sim_s_all, dim=1)
    log_all_e = torch.logsumexp(sim_e_all, dim=1)
    
    term_s = (1 - h) * (log_pos_s - log_all_s) + h * (log_neg_s - log_all_s)
    term_e = (1 - h) * (log_pos_e - log_all_e) + h * (log_neg_e - log_all_e)
    #term_s = (1 - h) * (log_pos_s - log_all_s)
    #term_e = (1 - h) * (log_pos_e - log_all_e)

    return -(term_s + term_e)


def margin_loss(sim_s_p, sim_s_m, sim_e_p, sim_e_m, h, margin: float = 0.2):
    max_p_s = sim_s_p.max(dim=1).values  # (1,)
    max_m_s = sim_s_m.max(dim=1).values  # (1,)
    max_p_e = sim_e_p.max(dim=1).values  # (1,)
    max_m_e = sim_e_m.max(dim=1).values  # (1,)
    min_p_s = sim_s_p.min(dim=1).values  # (1,)
    min_m_s = sim_s_m.min(dim=1).values  # (1,)
    min_p_e = sim_e_p.min(dim=1).values  # (1,)
    min_m_e = sim_e_m.min(dim=1).values  # (1,)

    loss_p_s = F.relu(margin - (max_p_s - max_m_s))
    loss_p_e = F.relu(margin - (max_p_e - max_m_e))
    loss_n_s = F.relu(margin - (max_m_s - max_p_s))
    loss_n_e = F.relu(margin - (max_m_e - max_p_e))
    #loss_p_s = F.relu(margin - (min_p_s - max_m_s))
    #loss_p_e = F.relu(margin - (min_p_e - max_m_e))
    #loss_n_s = F.relu(margin - (min_m_s - max_p_s))
    #loss_n_e = F.relu(margin - (min_m_e - max_p_e))
    loss = (1-h) * (loss_p_s + loss_p_e)  + h * (loss_n_s + loss_n_e)

    return loss




# ======================
# 2) 推論時のスコア計算（top-k span & sims）
# ======================

def contrastive_span_scores(
    mask_s_output,
    mask_e_output,
    labels,
    y_s_plus,
    y_s_minus,
    y_e_plus,
    y_e_minus,
    masked_span_output=None,   # (B,H)
    top_k: int = 5,
    max_span_len: int = 10,
    normalize: bool = True,
    sentence_ids = None,
):
    """
    推論・解析用:
      - doc側の candidate span の score(top_k)
      - masked_span_output との類似度(top_k)
    を返す。
    loss 計算もしたければ label を渡す（labels=None なら loss=None）。
    """
    B, H = mask_s_output.shape
    device = mask_s_output.device

    losses = []
    all_top_scores = []  # (B, top_k)
    all_top_sims = []    # (B, top_k)
    all_span_token_idx = []  # どのspanが予測されるか

    for i in range(B):
        sim_s_p, sim_s_m, sim_e_p, sim_e_m, sim_s_all, sim_e_all = _prepare_sims_for_one(
            mask_s_output[i],
            mask_e_output[i],
            y_s_plus[i],
            y_s_minus[i],
            y_e_plus[i],
            y_e_minus[i],
            device=device,
            normalize=normalize,
        )

        # ----- span 探索部分（元の推論コードをベースに） -----
        if sentence_ids is not None:
            sent_ids_i = sentence_ids[i]
            if isinstance(sent_ids_i, torch.Tensor):
                sent_ids_i = sent_ids_i.cpu().tolist()
        else:
            sent_ids_i = None
        #N = len(sent_ids_i) if sent_ids_i is not None else sim_s_m.size(1)
        
        sim_s = sim_s_m.squeeze(0)  # (N,)
        sim_e = sim_e_m.squeeze(0)  # (N,)
        N = sim_s.size(0)

        prefix_best_start = torch.zeros(N, dtype=torch.long, device=sim_s.device)
        best_idx = 0

        candidate_scores = torch.empty(N, device=device)
        candidate_s_idx = torch.empty(N, dtype=torch.long, device=device)
        candidate_e_idx = torch.empty(N, dtype=torch.long, device=device)

        for e in range(N):
            if sent_ids_i is not None:
                sent_e = sent_ids_i[e] # 文のID
            else:
                sent_e = None
            # ★ e を終端とする span の start は、この範囲に限定
            start_min = max(0, e - max_span_len + 1)
            # 同じsentence内に限定
            if sent_ids_i is not None:
                while start_min < e and sent_ids_i[start_min] != sent_e:
                    start_min += 1
            best_s = start_min
            best_score = sim_s[start_min] + sim_e[e]

            # start_min+1 .. e の範囲だけを探索
            for s in range(start_min + 1, e + 1):
                if sent_ids_i is not None and sent_ids_i[s] != sent_e:
                    continue
                score = sim_s[s] + sim_e[e]
                if score > best_score:
                    best_score = score
                    best_s = s

            if best_s is None:
                candidate_scores[e] = float("-inf")
                candidate_s_idx[e] = 0
                candidate_e_idx[e] = 0
            else:
                candidate_scores[e] = torch.exp(sim_s[best_s]) + torch.exp(sim_e[e])
                candidate_s_idx[e] = best_s
                candidate_e_idx[e] = e

        k = min(top_k, N)
        top_scores_i, top_indices = torch.topk(candidate_scores, k=k, largest=True)

        padded_scores = torch.full((top_k,), float("-inf"), device=device)
        padded_sims = torch.zeros(top_k, device=device)

        # ys_m 自体の埋め込みが欲しい場合、呼び出し側で保持しておく方が安全だが、
        # ここでは元コードに合わせて y_s_minus[i] からもう一度 tensor 化するなど設計し直してもOK

        # ここでは簡略化のため、上で使った ys_m を再構築
        ys_m_tokens = y_s_minus[i]
        if isinstance(ys_m_tokens, list):
            ys_m_tokens = torch.stack(ys_m_tokens, dim=0).to(device)

        if normalize:
            ys_m_tokens = F.normalize(ys_m_tokens, dim=-1)
            
        spans_for_example = [] # この事例に対して予測された span のリスト

        for rank, idx in enumerate(top_indices):
            s_idx = candidate_s_idx[idx].item()
            e_idx = candidate_e_idx[idx].item()
            
            if candidate_scores[idx] == float("-inf"):
                spans_for_example.append([])
                padded_scores[rank] = float("-inf")
                padded_sims[rank] = 0.0
                continue

            span_token_idx = list(range(s_idx, e_idx + 1))  # 予測された span のトークンインデックス
            spans_for_example.append(span_token_idx)
            
            # docの埋め込みから span ベクトルを計算
            doc_span_vecs = ys_m_tokens[s_idx : e_idx + 1]  # (L_span, H)
            doc_best_vec = doc_span_vecs.mean(dim=0)        # (H,)

            padded_scores[rank] = top_scores_i[rank]

            if masked_span_output is not None:
                z = masked_span_output[i]  # (H,)
                if normalize:
                    z = F.normalize(z.unsqueeze(0), dim=-1).squeeze(0)
                sim_span = torch.dot(doc_best_vec, z) 
                padded_sims[rank] = sim_span
            else:
                padded_sims[rank] = 0.0

        all_top_scores.append(padded_scores)
        all_top_sims.append(padded_sims)
        all_span_token_idx.append(spans_for_example) # 追加

        # ----- loss (オプション) -----
        if labels is not None:
            h = labels[i].float()
            log_pos_s = torch.logsumexp(sim_s_p, dim=1)
            log_neg_s = torch.logsumexp(sim_s_m, dim=1)
            log_all_s = torch.logsumexp(sim_s_all, dim=1)
            log_pos_e = torch.logsumexp(sim_e_p, dim=1)
            log_neg_e = torch.logsumexp(sim_e_m, dim=1)
            log_all_e = torch.logsumexp(sim_e_all, dim=1)

            term_s = (1 - h) * (log_pos_s - log_all_s) + 2 * h * (log_neg_s - log_all_s)
            term_e = (1 - h) * (log_pos_e - log_all_e) + 2 * h * (log_neg_e - log_all_e)
            loss_i = -(term_s + term_e)
            losses.append(loss_i.squeeze(0))

    score_tensor = torch.stack(all_top_scores)   # (B, top_k)
    sim_tensor = torch.stack(all_top_sims)       # (B, top_k)
    loss = torch.stack(losses).mean() if (labels is not None and len(losses) > 0) else None

    return loss, score_tensor, sim_tensor, all_span_token_idx
