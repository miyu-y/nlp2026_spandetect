# utis/model.py
import os
from typing import List, Optional, Tuple
import random

import torch
import torch.nn as nn
from transformers.modeling_outputs import ModelOutput

random.seed(42)


class HalNPMBase(nn.Module):
    """
    NPMベースモデルの共通部分。
    - base_model: facebook/npm などの encoder
    - loss_function: contrastive_loss 系（train用 or infer用）を注入
    """

    def __init__(
        self,
        base_model: nn.Module,
        loss_function,
        tokenizer,
        question_encoder: Optional[nn.Module] = None,
        generator: Optional[nn.Module] = None,
        dropout_prob: float = 0.1,
        loss_mode: str = "contrastive",
    ):
        super().__init__()
        self.base_model = base_model
        self.dropout = nn.Dropout(dropout_prob)
        self.loss_function = loss_function
        self.question_encoder = question_encoder
        self.generator = generator
        self.tokenizer = tokenizer
        self.loss_mode = loss_mode

        self.mask_token_id = tokenizer.mask_token_id
        self.hidden_size = base_model.config.hidden_size

    # ---------- 共通ヘルパ ----------

    def _encode_doc_and_text(
        self,
        doc_input_ids,
        text_input_ids,
        doc_attention_mask,
        text_attention_mask,
    ):
        """doc / text をそれぞれ encode して (B, L, H) を返す"""
        doc_output = self.base_model(
            input_ids=doc_input_ids,
            attention_mask=doc_attention_mask,
            return_dict=True,
        )[0]
        text_output = self.base_model(
            input_ids=text_input_ids,
            attention_mask=text_attention_mask,
            return_dict=True,
        )[0]

        doc_output = self.dropout(doc_output)
        text_output = self.dropout(text_output)
        return doc_output, text_output

    def _extract_mask_vectors_single(self, text_input_ids, text_output):
        """
        text 側から <mask> 2つの hidden を取る。
        前提：1サンプルにつき <mask> が2kつ。
        """
        device = text_input_ids.device
        batch_size = text_input_ids.size(0)
        mask_pos = (text_input_ids == self.mask_token_id).nonzero(as_tuple=False)
        mask_pos = mask_pos.view(batch_size, 2, 2)  # (B, 2, 2: [b, tok_idx])
        batch_idx = torch.arange(batch_size, device=device)
        mask_s_idx = mask_pos[:, 0, 1]  # 最初の <mask>
        mask_e_idx = mask_pos[:, 1, 1]  # 2番目の <mask>
        mask_s_output = text_output[batch_idx, mask_s_idx]  # (B,H)
        mask_e_output = text_output[batch_idx, mask_e_idx]  # (B,H)
        return mask_s_output, mask_e_output

    def _extract_mask_vectors_multi(self, text_input_ids, text_output):
        """
        text 側から <mask> の hidden を取る（multi-span対応）。

        前提：
        - 1サンプルにつき <mask> が 2*k 個（start/end のペア）
        - 各サンプルで k は異なってよい
        返り値：
        mask_s_output: (B, max_k, H)
        mask_e_output: (B, max_k, H)
        valid_k:       (B,)  各サンプルのk（num_spansが無い場合は推定）
        """
        device = text_input_ids.device
        B, T, H = text_output.shape

        # (b, t) のmask位置を全部集める
        mask_pos = (text_input_ids == self.mask_token_id).nonzero(as_tuple=False)  # (N_mask_total, 2)

        # サンプルごとに mask index のリストを作る
        mask_indices_per_sample = [[] for _ in range(B)]
        for b, t in mask_pos.tolist():
            mask_indices_per_sample[b].append(t)

        # 各サンプル内のmask位置を昇順に（テキスト順）
        for b in range(B):
            mask_indices_per_sample[b].sort()

        # k の決定
        valid_k = torch.tensor(
            [len(mask_indices_per_sample[b]) // 2 for b in range(B)],
            device=device,
            dtype=torch.long,
        )

        max_k = int(valid_k.max().item()) if B > 0 else 0

        # 出力を pad 付きで用意
        # pad 部分はゼロベクトル（後段で valid_k を見て無視）
        mask_s_output = torch.zeros((B, max_k, H), device=device, dtype=text_output.dtype)
        mask_e_output = torch.zeros((B, max_k, H), device=device, dtype=text_output.dtype)

        for b in range(B):
            idxs = mask_indices_per_sample[b]
            # 期待する mask 個数は 2*k
            k = int(valid_k[b].item())
            need = 2 * k

            # 足りない／余る場合は安全に対処（データ不整合の検知にもなる）
            if len(idxs) < need:
                # 足りない分は無視（または raise でもOK）
                # raise ValueError(f"sample {b}: masks {len(idxs)} < expected {need}")
                k = len(idxs) // 2
                need = 2 * k
            elif len(idxs) > need:
                # 余分がある場合：先頭から使う（基本起きないはず）
                idxs = idxs[:need]

            if k == 0:
                continue

            # ペア化： (s0,e0,s1,e1,...) を (k,2) に
            # ここは「mask_text_multi が必ず start→end の順で挿入している」前提
            pair = torch.tensor(idxs, device=device, dtype=torch.long).view(k, 2)
            s_idx = pair[:, 0]
            e_idx = pair[:, 1]

            mask_s_output[b, :k, :] = text_output[b, s_idx, :]
            mask_e_output[b, :k, :] = text_output[b, e_idx, :]

        return mask_s_output, mask_e_output, valid_k

    def _build_y_sets_train(
        self,
        doc_output: torch.Tensor,  # (B, L, H)
        ngram_token_start,  # List[List[List[int]]]  (B, k, m_l)
        ngram_token_end,  # List[List[List[int]]]  (B, k, m_l)
        doc_attention_mask: torch.Tensor,  # (B, L)
        valid_k: torch.Tensor,  # (B,)
    ):
        """
        訓練時（multi-span）:
        - 事例 i に mask が k 個
        - mask l ごとに、doc 内の一致候補 start/end が複数(m_l)ある
        - それを正例集合として Y+ を作り、それ以外（かつ attention_mask=1）を Y- に入れる
        返り値は span 単位に flatten した list。
        """
        device = doc_output.device
        B, L, H = doc_output.size()

        zero_vec = torch.zeros(self.hidden_size, device=device)

        y_s_plus_flat, y_s_minus_flat = [], []
        y_e_plus_flat, y_e_minus_flat = [], []

        for i in range(B):
            k_i = int(valid_k[i].item())

            # safety: ngram が None の場合
            starts_i = ngram_token_start[i] if ngram_token_start is not None else []
            ends_i = ngram_token_end[i] if ngram_token_end is not None else []

            for l in range(k_i):
                # 候補リスト（token idx の list）
                s_list = starts_i[l] if l < len(starts_i) and starts_i[l] is not None else []
                e_list = ends_i[l] if l < len(ends_i) and ends_i[l] is not None else []

                # None 除去 & 範囲外除去
                s_pos = [int(x) for x in s_list if x is not None and 0 <= int(x) < L]
                e_pos = [int(x) for x in e_list if x is not None and 0 <= int(x) < L]

                start_set = set(s_pos)
                end_set = set(e_pos)

                y_s_p, y_s_m = [], []
                y_e_p, y_e_m = [], []

                for j in range(L):
                    if doc_attention_mask[i, j].item() == 0:  # padding
                        continue

                    if j in start_set:
                        y_s_p.append(doc_output[i, j])
                    else:
                        y_s_m.append(doc_output[i, j])

                    if j in end_set:
                        y_e_p.append(doc_output[i, j])
                    else:
                        y_e_m.append(doc_output[i, j])

                # 空対策
                if len(y_s_p) == 0:
                    y_s_p.append(zero_vec)
                if len(y_s_m) == 0:
                    y_s_m.append(zero_vec)
                if len(y_e_p) == 0:
                    y_e_p.append(zero_vec)
                if len(y_e_m) == 0:
                    y_e_m.append(zero_vec)

                # y_minusをカット

                # sample_len = min(len(y_s_p) * 50, len(y_s_m)//4)
                # y_s_m = random.sample(y_s_m, sample_len)
                # y_e_m = random.sample(y_e_m, sample_len)

                # flatten に追加
                y_s_plus_flat.append(y_s_p)
                y_s_minus_flat.append(y_s_m)
                y_e_plus_flat.append(y_e_p)
                y_e_minus_flat.append(y_e_m)

        return y_s_plus_flat, y_s_minus_flat, y_e_plus_flat, y_e_minus_flat

    def _build_y_sets_infer(
        self,
        doc_output: torch.Tensor,  # (B, L, H)
        doc_attention_mask: torch.Tensor,  # (B, L)
    ):
        """
        推論時:
        - 全トークンを minus 側に入れるが、
            attention_mask == 0 (padding) は除外する。
        - plus 側はダミーのゼロベクトルだけ。
        """
        device = doc_output.device
        B, L, H = doc_output.size()

        zero_vec = torch.zeros(self.hidden_size, device=device)

        # plus は全部ダミーでOK
        y_s_plus = [[zero_vec] for _ in range(B)]
        y_e_plus = [[zero_vec] for _ in range(B)]

        y_s_minus = []
        y_e_minus = []

        for i in range(B):
            minus_tokens = []
            # attention_mask を見て pad をスキップ
            for j in range(L):
                if doc_attention_mask[i, j] == 0:
                    continue  # ★ padding は入れない
                minus_tokens.append(doc_output[i, j])

            # もし全部 pad で何も残らなかった場合の保険
            if len(minus_tokens) == 0:
                minus_tokens.append(zero_vec)

            y_s_minus.append(minus_tokens)
            y_e_minus.append(minus_tokens)

        return y_s_plus, y_s_minus, y_e_plus, y_e_minus

    def _encode_masked_span(
        self,
        masked_span_input_ids,
        masked_span_attention_mask,
        masked_span_index: Optional[List[Tuple[int, int]]] = None,
    ):
        """
        original_text から masked span 部分の埋め込みを作る（推論用）。
        - masked_span_index があれば [s,e) の区間平均
        - なければ文全体の attention-weighted 平均
        """
        if masked_span_input_ids is None:
            return None

        device = masked_span_input_ids.device
        out = self.base_model(
            input_ids=masked_span_input_ids,
            attention_mask=masked_span_attention_mask,
            return_dict=True,
        )[0]
        out = self.dropout(out)
        B, L, H = out.size()

        if masked_span_index is not None:  # text全体の埋め込みからの取り出し
            vecs = []
            for i in range(B):
                s, e = masked_span_index[i]
                span_vec = out[i, s:e, :]
                attn = masked_span_attention_mask[i, s:e].unsqueeze(-1)  # (L_span,1)
                span_vec = (span_vec * attn).sum(dim=0) / attn.sum()
                vecs.append(span_vec)
            masked_span_output = torch.stack(vecs, dim=0)  # (B,H)
        else:  # masked span 単体の埋め込み
            attn = masked_span_attention_mask.unsqueeze(-1)  # (B,L,1)
            masked_span_output = (out * attn).sum(dim=1) / attn.sum(dim=1)  # (B,H)

        return masked_span_output

    def _pack_scores(self, score_flat: torch.Tensor, valid_k: torch.Tensor, pad_value: float = -1e9):
        """
        score_flat: (total_spans,)
        valid_k: (B,)
        return: (B, max_k) pad済み
        """
        device = score_flat.device
        B = valid_k.size(0)
        max_k = int(valid_k.max().item()) if B > 0 else 1
        if max_k == 0:
            max_k = 1

        out = torch.full((B, max_k), pad_value, device=device, dtype=score_flat.dtype)

        cursor = 0
        for b in range(B):
            k = int(valid_k[b].item())
            if k > 0:
                out[b, :k] = score_flat[cursor : cursor + k]
                cursor += k

        return out

    # ---------- save / load 共通 ----------

    def save_pretrained(self, save_directory: str):
        os.makedirs(save_directory, exist_ok=True)
        if self.question_encoder is not None:
            self.question_encoder.save_pretrained(os.path.join(save_directory, "question_encoder"))
        if self.generator is not None:
            self.generator.save_pretrained(os.path.join(save_directory, "generator"))
        torch.save(self.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))

    @classmethod
    def from_pretrained(
        cls,
        base_model: nn.Module,
        loss_function,
        tokenizer,
        save_directory: str,
        **kwargs,
    ):
        from transformers import AutoModel  # 遅延 import でも可

        question_encoder = None
        generator = None

        q_path = os.path.join(save_directory, "question_encoder")
        g_path = os.path.join(save_directory, "generator")

        if os.path.exists(q_path):
            question_encoder = AutoModel.from_pretrained(q_path)
        if os.path.exists(g_path):
            generator = AutoModel.from_pretrained(g_path)

        model = cls(
            base_model=base_model,
            loss_function=loss_function,
            tokenizer=tokenizer,
            question_encoder=question_encoder,
            generator=generator,
        )
        state_dict = torch.load(
            os.path.join(save_directory, "pytorch_model.bin"),
            map_location="cpu",
        )
        model.load_state_dict(state_dict)

        # 追加のキーワード引数（top_k 等）をモデル属性としてセット
        for k, v in kwargs.items():
            setattr(model, k, v)

        return model


# =======================
# 訓練用モデル
# =======================


class HalNPMTrainModel(HalNPMBase):
    """
    Trainer で NPM を追加学習するときに使うモデル。
    forward は (loss, logits) を返す。
    """

    def forward(
        self,
        input_ids,
        attention_mask=None,
        labels=None,
        ngram_token_start=None,
        ngram_token_end=None,
        return_dict=None,
        **kwargs,
    ):
        # input_ids: [doc_ids, text_ids]
        doc_input_ids = input_ids[0]
        text_input_ids = input_ids[1]
        doc_attention_mask = attention_mask[0]
        text_attention_mask = attention_mask[1]

        B = doc_input_ids.size(0)

        doc_output, text_output = self._encode_doc_and_text(
            doc_input_ids, text_input_ids, doc_attention_mask, text_attention_mask
        )

        mask_s_output, mask_e_output, valid_k = self._extract_mask_vectors_multi(text_input_ids, text_output)

        mask_s_flat = torch.cat([mask_s_output[i, : valid_k[i]] for i in range(B)], dim=0)  # (total_spans,H)
        mask_e_flat = torch.cat([mask_e_output[i, : valid_k[i]] for i in range(B)], dim=0)
        labels_flat = torch.cat(
            [labels[i, : valid_k[i]] for i in range(B)],
            dim=0,
        )  # (total_spans,)

        # ngram_token_start / end は各サンプルごとの「正例 token index リスト」を想定
        y_s_plus, y_s_minus, y_e_plus, y_e_minus = self._build_y_sets_train(
            doc_output, ngram_token_start, ngram_token_end, doc_attention_mask, valid_k
        )

        loss, score_list = self.loss_function(
            mask_s_flat,  # mask_s_output
            mask_e_flat,  # mask_e_output
            labels_flat,  # labels
            y_s_plus,
            y_s_minus,
            y_e_plus,
            y_e_minus,
            mode=self.loss_mode,  # 追加
        )
        score_padded = self._pack_scores(score_list, valid_k, pad_value=-1e9)
        return ModelOutput(loss=loss, logits=score_padded)


# =======================
# 推論・解析用モデル
# =======================


class HalNPMInferenceModel(HalNPMBase):
    """
    推論時に:
      - spanごとの score (top-k)
      - 元スパンとの類似度
    を出したいときに使うモデル。
    """

    def forward(
        self,
        input_ids,
        attention_mask=None,
        labels=None,
        masked_span_index=None,
        ngram_token_start=None,
        ngram_token_end=None,
        return_dict=None,
        top_k: int = 5,
        sentence_ids=None,
        masked_span_output=None,
        **kwargs,
    ):
        # input_ids: [doc_ids, text_ids, masked_span_ids?]
        doc_input_ids = input_ids[0]
        text_input_ids = input_ids[1]
        masked_span_input_ids = input_ids[2] if len(input_ids) > 2 else None

        doc_attention_mask = attention_mask[0]
        text_attention_mask = attention_mask[1]
        masked_span_attention_mask = (
            attention_mask[2] if attention_mask is not None and len(attention_mask) > 2 else None
        )

        device = doc_input_ids.device
        batch_size = doc_input_ids.size(0)

        # ngram_token_start/end が Tensor で来た場合に int に直す（元コード踏襲）
        if isinstance(ngram_token_start, torch.Tensor):
            ngram_token_start = [int(x) if x.item() >= 0 else None for x in ngram_token_start]
        if isinstance(ngram_token_end, torch.Tensor):
            ngram_token_end = [int(x) if x.item() >= 0 else None for x in ngram_token_end]

        doc_output, text_output = self._encode_doc_and_text(
            doc_input_ids, text_input_ids, doc_attention_mask, text_attention_mask
        )

        mask_s_output, mask_e_output = self._extract_mask_vectors_single(text_input_ids, text_output)

        # 推論時は全部 minus に入れる
        y_s_plus, y_s_minus, y_e_plus, y_e_minus = self._build_y_sets_infer(doc_output, doc_attention_mask)

        # print(len(self.tokenizer.convert_ids_to_tokens(doc_input_ids[0].cpu().tolist())), len(y_s_minus[0]))

        if masked_span_output is None and masked_span_input_ids is not None:
            masked_span_output = self._encode_masked_span(
                masked_span_input_ids,
                masked_span_attention_mask,
                masked_span_index=masked_span_index,
            )

        loss, score_list, sim_list, span_token_index = self.loss_function(
            mask_s_output,
            mask_e_output,
            labels,
            y_s_plus,
            y_s_minus,
            y_e_plus,
            y_e_minus,
            masked_span_output=masked_span_output,
            top_k=top_k,
            sentence_ids=sentence_ids,  ##
        )

        return ModelOutput(loss=loss, logits=score_list, similarity=sim_list, span_token_index=span_token_index)  # 追加
