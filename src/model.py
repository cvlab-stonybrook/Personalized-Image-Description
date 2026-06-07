import os
import copy
import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
from qwen_vl_utils import process_vision_info

from visual_utils import visualize_features_with_fixations
from model_utils import (SelfAttentionLayer, CrossAttentionLayer, get_duration_positional_encoding,
                         FFNLayer, PositionalEncoding2D, PositionalEncoding1D)
from transformers import AutoImageProcessor, AutoModel
from typing import Dict, Any, Optional

from utils import print_rank0, _strip_assistant_turns, save_prediction

class ScanAutoEncoder(nn.Module):
    """
    Encoder:
      - Context = image_feat + caption_feat
      - Refine scan_feat with cross-attn (query=scan_feat, memory=context)
      - Produce two latents: z_subj (personality), z_scan (scanpath latent)
    Decoder:
      - Build T queries simultaneously using learned time embeddings + fused latent bias
      - Decode with self-attn (non-causal) + cross-attn to latent + cross-attn to context
      - Output (B, T, 3) = (x, y, v) in [0,1], where v is validity of the box

    Also classifies subject from z_subj (personality).
    """
    def __init__(self, qwen_model, processor, device, hparams):
        super().__init__()
        D = hparams.sen_hidden_size
        self.device = device
        self.processor = processor
        self.hparams = hparams

        # -------------------- Backbones --------------------
        img_encoder_name = hparams.img_encoder_name
        self.image_encoder = AutoModel.from_pretrained(img_encoder_name)
        for p in self.image_encoder.parameters():
            p.requires_grad = False

        self.image_processor = AutoImageProcessor.from_pretrained(
            img_encoder_name, do_resize=False, do_center_crop=False
        )

        self.text_encoder = copy.deepcopy(qwen_model.get_input_embeddings())
        for p in self.text_encoder.parameters():
            p.requires_grad = False

        # -------------------- Positional encodings --------------------
        self.spatial_pos_encoder = PositionalEncoding2D(
            d_model=D, max_H=hparams.feat_h // 2, max_W=hparams.feat_w // 2
        )
        self.order_pos_encoder = PositionalEncoding1D(d_model=D, max_len=2048)

        # -------------------- Projections --------------------
        self.img_feat_proj  = nn.Linear(hparams.img_feat_dim,     D)
        self.cap_feat_proj  = nn.Linear(hparams.qwen_hidden_size, D)
        self.scan_feat_proj = nn.Linear(hparams.img_feat_dim,     D)

        # -------------------- Encoder: refine scan_feat with context --------------------
        enc_layers = getattr(hparams, "scan_enc_layers", 2)
        self.enc_scan_self  = nn.ModuleList()
        self.enc_scan_cross = nn.ModuleList()
        self.enc_scan_ffn   = nn.ModuleList()
        self.enc_cap_self   = nn.ModuleList()
        self.enc_cap_cross  = nn.ModuleList()
        self.enc_cap_ffn    = nn.ModuleList()
        for _ in range(enc_layers):
            self.enc_scan_self.append(
                SelfAttentionLayer(d_model=D, nhead=hparams.nhead, dropout=hparams.dropout, normalize_before=False)
            )
            self.enc_scan_cross.append(
                CrossAttentionLayer(d_model=D, nhead=hparams.nhead, dropout=hparams.dropout, normalize_before=False)
            )
            self.enc_scan_ffn.append(
                FFNLayer(d_model=D, dim_feedforward=D, dropout=hparams.dropout, normalize_before=False)
            )
            # tag
            self.enc_cap_self.append(
                SelfAttentionLayer(d_model=D, nhead=hparams.nhead, dropout=hparams.dropout, normalize_before=False)
            )
            self.enc_cap_cross.append(
                CrossAttentionLayer(d_model=D, nhead=hparams.nhead, dropout=hparams.dropout, normalize_before=False)
            )
            self.enc_cap_ffn.append(
                FFNLayer(d_model=D, dim_feedforward=D, dropout=hparams.dropout, normalize_before=False)
            )

        # -------------------- Latent heads: z_subj & z_scan --------------------
        # Multi-token personality probe for z_subj, single probe for z_scan
        self.subj_query_tokens = nn.Embedding(getattr(hparams, "num_subj_tokens", 1), D)
        self.scan_query_token  = nn.Parameter(torch.randn(1, 1, D))

        self.pool_subj_cross = CrossAttentionLayer(d_model=D, nhead=hparams.nhead, dropout=hparams.dropout, normalize_before=False)
        self.pool_scan_cross = CrossAttentionLayer(d_model=D, nhead=hparams.nhead, dropout=hparams.dropout, normalize_before=False)

        self.z_subj_mlp = nn.Sequential(
            nn.LayerNorm(D), nn.Linear(D, D), nn.GELU(), nn.Linear(D, D)
        )
        self.z_scan_mlp = nn.Sequential(
            nn.LayerNorm(D), nn.Linear(D, D), nn.GELU(), nn.Linear(D, D)
        )

        # -------------------- Subject classifier (from z_subj) --------------------
        self.classifier = nn.Linear(D, hparams.num_subjects)

        # -------------------- Parallel Decoder (no teacher forcing) --------------------
        # Time embeddings for T positions
        self.max_len  = getattr(hparams, "scan_max_len", 512)
        self.time_embed = nn.Embedding(self.max_len, D)

        # Fuse z_subj and z_scan to provide latent memory and bias
        self.latent_fuse = nn.Sequential(
            nn.LayerNorm(D * 2),
            nn.Linear(D * 2, D),
            nn.GELU(),
            nn.Linear(D, D),
        )

        dec_layers = getattr(hparams, "scan_dec_layers", 4)
        self.dec_self   = nn.ModuleList()
        self.dec_latent = nn.ModuleList()
        self.dec_ctx    = nn.ModuleList()
        self.dec_ffn    = nn.ModuleList()
        for _ in range(dec_layers):
            self.dec_self.append(  # non-causal: produce all steps simultaneously
                SelfAttentionLayer(d_model=D, nhead=hparams.nhead, dropout=hparams.dropout, normalize_before=False)
            )
            self.dec_latent.append(
                CrossAttentionLayer(d_model=D, nhead=hparams.nhead, dropout=hparams.dropout, normalize_before=False)
            )
            self.dec_ctx.append(
                CrossAttentionLayer(d_model=D, nhead=hparams.nhead, dropout=hparams.dropout, normalize_before=False)
            )
            self.dec_ffn.append(
                FFNLayer(d_model=D, dim_feedforward=D, dropout=hparams.dropout, normalize_before=False)
            )

        # Output head: predict (x, y, v) per step
        self.scan_out = nn.Linear(D, 5)
        self.valid_threshold = getattr(hparams, "scan_valid_thresh", 0.5)

        self.subj_adapter = nn.Sequential(
            nn.LayerNorm(D),
            nn.Linear(D, 2 * len(self.dec_self) * D),  # gamma,beta per layer
        )

    # ==================== helpers ====================
    def _build_parallel_queries(self, B: int, T: int, fused_latent: torch.Tensor) -> torch.Tensor:
        """
        Build decoder query tokens for all steps at once:
          - time embedding for steps [0..T-1]
          - add a broadcast latent bias from fused_latent
        returns queries of shape (T, B, D)
        """
        # (T, D)
        t_idx = torch.arange(T, device=self.device).clamp(max=self.max_len - 1)
        time_q = self.time_embed(t_idx)  # (T, D)

        # fused_latent: (B, D) -> (T, B, D)
        latent_bias = fused_latent.unsqueeze(0).expand(T, B, -1)
        queries = time_q.unsqueeze(1).expand(-1, B, -1) + latent_bias  # (T, B, D)
        return queries

    def _decoder_parallel(self,
                          z_subj: torch.Tensor,
                          queries: torch.Tensor,
                          ctx_feat: torch.Tensor, ctx_mask: torch.Tensor,
                          fused_latent: torch.Tensor) -> torch.Tensor:
        """
        queries: (T,B,D) decoder queries for all steps
        ctx_feat: (L_ctx,B,D), ctx_mask: (B,L_ctx)
        fused_latent: (B,D) → memory token (1,B,D)
        return: (T,B,D)
        """
        B = fused_latent.size(0)
        latent_mem  = fused_latent.unsqueeze(0)  # (1,B,D)
        latent_mask = torch.zeros(B, 1, device=self.device, dtype=torch.bool)

        x = queries

        B, D = z_subj.size()
        L = len(self.dec_self)  # number of decoder blocks
        gamma_beta = self.subj_adapter(z_subj)             # (B, 2*L*D)
        gamma, beta = gamma_beta.split(L*D, dim=-1)        # (B, L*D) each
        gamma = gamma.view(B, L, D)                        # (B, L, D)
        beta  = beta.view(B, L, D)                         # (B, L, D)

        for i in range(len(self.dec_self)):
            x = self.dec_self[i](tgt=x)[0]  # non-causal self-attn over time
            x = self.dec_latent[i](tgt=x, memory=latent_mem, memory_key_padding_mask=latent_mask)[0]
            x = self.dec_ctx[i](tgt=x, memory=ctx_feat, memory_key_padding_mask=ctx_mask)[0]

            # ---- FiLM from z_subj for layer i ----  
            g = torch.tanh(gamma[:, i, :]).unsqueeze(0)  # (1, B, D)
            b = torch.tanh(beta[:,  i, :]).unsqueeze(0)  # (1, B, D)
            x = x * (1.0 + g) + b                        # (T, B, D)   # tag

            x = self.dec_ffn[i](x)

        return x  # (T,B,D)


    def _roi_pool_tokens_xyxy(self,
                          img_tokens: torch.Tensor,  # (B, L_I, D_img) with L_I = H_out * W_out
                          bboxes_xyxy: torch.Tensor, # (B, T, 4) in [0,1]
                          H_out: int, W_out: int) -> torch.Tensor:
        """
        Mean-pool ViT patch tokens that lie inside each bbox (x1,y1,x2,y2), all normalized to [0,1].
        Returns (B, T, D_img). Falls back to nearest single token if empty.
        """
        B, L_I, Dimg = img_tokens.shape
        T = bboxes_xyxy.size(1)

        # grid coordinates for each token index, flattened
        ys = torch.arange(H_out, device=img_tokens.device)
        xs = torch.arange(W_out, device=img_tokens.device)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")  # (H_out, W_out)
        flat_x = xx.flatten()[None, None, :]            # (1,1,L_I)
        flat_y = yy.flatten()[None, None, :]            # (1,1,L_I)

        # bbox corners in token grid coords
        x1 = (bboxes_xyxy[..., 0] * W_out).unsqueeze(-1)
        y1 = (bboxes_xyxy[..., 1] * H_out).unsqueeze(-1)
        x2 = (bboxes_xyxy[..., 2] * W_out).unsqueeze(-1)
        y2 = (bboxes_xyxy[..., 3] * H_out).unsqueeze(-1)

        # inclusive integer bounds for tokens
        x1i = x1.floor()
        y1i = y1.floor()
        x2i = (x2.ceil() - 1).clamp_min_(x1i)  # ensure x2i >= x1i
        y2i = (y2.ceil() - 1).clamp_min_(y1i)

        in_x = (flat_x >= x1i) & (flat_x <= x2i)
        in_y = (flat_y >= y1i) & (flat_y <= y2i)
        mask = (in_x & in_y)                      # (B,T,L_I)

        # handle empty selections: use nearest token to bbox center
        empty = mask.sum(-1) == 0                 # (B,T)
        if empty.any():
            cx = ((x1 + x2) * 0.5).round().clamp_(0, W_out - 1)
            cy = ((y1 + y2) * 0.5).round().clamp_(0, H_out - 1)
            nn_idx = (cy * W_out + cx).long()     # (B,T,1)
            onehot = torch.zeros_like(mask, dtype=mask.dtype)
            onehot.scatter_(-1, nn_idx, True)
            mask = torch.where(empty.unsqueeze(-1), onehot, mask)

        weights = mask.float()
        weights = weights / weights.sum(-1, keepdim=True).clamp(min=1.0)
        # (B,T,L_I) @ (B,L_I,Dimg) -> (B,T,Dimg)
        pooled = torch.einsum("btl,bld->btd", weights, img_tokens)
        return pooled



    def _bbox_iou(self, pred, tgt):
        # pred/tgt: (..., 4) in (cx,cy,w,h) normalized to [0,1]
        # convert to (x1,y1,x2,y2)
        px1 = pred[...,0] - pred[...,2]/2; py1 = pred[...,1] - pred[...,3]/2
        px2 = pred[...,0] + pred[...,2]/2; py2 = pred[...,1] + pred[...,3]/2
        tx1 = tgt[...,0] - tgt[...,2]/2; ty1 = tgt[...,1] - tgt[...,3]/2
        tx2 = tgt[...,0] + tgt[...,2]/2; ty2 = tgt[...,1] + tgt[...,3]/2

        ix1 = torch.max(px1, tx1); iy1 = torch.max(py1, ty1)
        ix2 = torch.min(px2, tx2); iy2 = torch.min(py2, ty2)
        iw = (ix2 - ix1).clamp(min=0); ih = (iy2 - iy1).clamp(min=0)
        inter = iw * ih
        ap = (px2-px1).clamp(min=0) * (py2-py1).clamp(min=0)
        at = (tx2-tx1).clamp(min=0) * (ty2-ty1).clamp(min=0)
        union = ap + at - inter + 1e-6
        return inter / union

    def _outputs_and_loss(self,
                      dec_states: torch.Tensor,
                      bboxes_xyxy: torch.Tensor,  # (B,T,4) in [0,1]
                      mask: torch.Tensor):
        """
        dec_states: (T,B,D)
        returns recon with (x1,y1,x2,y2,v)
        """
        logits = self.scan_out(dec_states).transpose(0, 1)  # (B,T,5)

        # map to [0,1]; enforce x2>=x1, y2>=y1 by sorting pairs
        x1p = torch.sigmoid(logits[..., 0])
        y1p = torch.sigmoid(logits[..., 1])
        x2p = torch.sigmoid(logits[..., 2])
        y2p = torch.sigmoid(logits[..., 3])

        x_min = torch.minimum(x1p, x2p)
        x_max = torch.maximum(x1p, x2p)
        y_min = torch.minimum(y1p, y2p)
        y_max = torch.maximum(y1p, y2p)

        # tiny epsilon to avoid zero-area boxes
        eps = 1e-6
        x_max = (x_max - x_min).clamp(min=eps) + x_min
        y_max = (y_max - y_min).clamp(min=eps) + y_min

        v_hat = torch.sigmoid(logits[..., 4])

        pred = torch.stack([x_min, y_min, x_max, y_max], dim=-1)  # (B,T,4)
        valid = (~mask).float()                                   # (B,T)

        # Smooth L1 on xyxy (stable) over valid steps
        beta = 1/9
        l1 = torch.nn.functional.smooth_l1_loss(
            pred, bboxes_xyxy, beta=beta, reduction='none'
        ).sum(-1)                                                # (B,T)
        bbox_l1 = (l1 * valid).sum() / valid.sum().clamp(min=1.0)

        # (optional) IoU monitor
        with torch.no_grad():
            iou = self._iou_xyxy(pred, bboxes_xyxy)
            iou_mean = (iou * valid).sum() / valid.sum().clamp(min=1.0)

        # validity BCE
        v_tgt = valid
        v_bce = -(v_tgt * torch.log(v_hat.clamp(min=eps)) +
                (1.0 - v_tgt) * torch.log((1.0 - v_hat).clamp(min=eps)))
        v_bce = v_bce.mean()

        total = bbox_l1 + v_bce
        recon_scan = torch.cat([pred, v_hat.unsqueeze(-1)], dim=-1)  # (B,T,5)

        return {
            "recon": {"recon_scan": recon_scan.detach()},
            "losses": {"bbox_l1": bbox_l1, "v_bce": v_bce, "total": total, "iou_mean": iou_mean.detach()}
        }

    def _iou_xyxy(self, p: torch.Tensor, t: torch.Tensor):
        # p,t: (...,4) xyxy in [0,1]
        px1, py1, px2, py2 = p.unbind(-1)
        tx1, ty1, tx2, ty2 = t.unbind(-1)
        ix1 = torch.maximum(px1, tx1); iy1 = torch.maximum(py1, ty1)
        ix2 = torch.minimum(px2, tx2); iy2 = torch.minimum(py2, ty2)
        iw = (ix2 - ix1).clamp(min=0); ih = (iy2 - iy1).clamp(min=0)
        inter = iw * ih
        ap = (px2 - px1).clamp(min=0) * (py2 - py1).clamp(min=0)
        at = (tx2 - tx1).clamp(min=0) * (ty2 - ty1).clamp(min=0)
        return inter / (ap + at - inter + 1e-6)




    # ==================== forward ====================
    def forward(self,
                images,
                captions,
                scanpaths,
                image_names: Optional[Any] = None,
                *,
                reconstruct: bool = True) -> Dict[str, Any]:
        """
        images: list of PIL.Image or tensors
        captions: dict with 'input_ids', 'attention_mask'
        scanpaths: dict with 'X','Y','mask' (B,T); X,Y in [0,1], mask=True for PAD
        """
        # ---------- 1) Encode image ----------
        inputs = self.image_processor(images=images, return_tensors="pt").to(self.device)
        img_raw = self.image_encoder(**inputs).last_hidden_state[:, 1:, :]  # (B, L_I, D_img) w/o CLS
        # img_raw = self.image_encoder(**inputs).last_hidden_state[:, 5:, :]   # (B, L_I, D_img) w/o CLS
        B, L_I, D_img = img_raw.shape
        W_out = int(math.sqrt(L_I))
        H_out = W_out

        # ---------- 2) Caption token embeddings ----------
        input_ids = captions["input_ids"].to(self.device)
        cap_raw = self.text_encoder(input_ids)  # (B, L_C, D_qwen)
        cap_raw = cap_raw.to(self.img_feat_proj.weight.dtype)
        
        bboxes = scanpaths["bboxes"].to(self.device)          # (B,T,4)
        T = bboxes.size(1)

        # ROI-pool tokens within each bbox
        scan_raw = self._roi_pool_tokens_xyxy(img_raw, bboxes, H_out, W_out)  # (B,T,D_img)

        # centers for spatial positional encoding
        cx = 0.5 * (bboxes[..., 0] + bboxes[..., 2])
        cy = 0.5 * (bboxes[..., 1] + bboxes[..., 3])

        # ---------- 4) Project to shared D ----------
        img  = self.img_feat_proj(img_raw)     # (B,L_I,D)
        cap  = self.cap_feat_proj(cap_raw)     # (B,L_C,D)
        scan = self.scan_feat_proj(scan_raw)   # (B,T,D)

        # Positional encodings: center-only is fine
        px = (cx * W_out).long().clamp(0, W_out - 1)
        py = (cy * H_out).long().clamp(0, H_out - 1)
        # tag
        scan = scan + self.spatial_pos_encoder(px, py) + self.order_pos_encoder(scan)
        duration_encoding = get_duration_positional_encoding(scanpaths["T"].to(self.device), self.hparams.sen_hidden_size, self.device)  # (B,T,D)
        scan = scan + duration_encoding
        # To (L,B,D)
        img  = img.transpose(0, 1)     # (L_I,B,D)
        cap  = cap.transpose(0, 1)     # (L_C,B,D)
        scan = scan.transpose(0, 1)    # (T,B,D)

        #Masks
        cap_mask  = (captions["attention_mask"] == 0).to(self.device, dtype=torch.bool, non_blocking=True)  # (B,L_C)
        img_mask  = torch.zeros(img.size(1), img.size(0), dtype=torch.bool, device=self.device)             # (B,L_I) False
        scan_mask = scanpaths["mask"].to(self.device)  # (B,T) True=PAD

        # ---------- 5) Context ----------
        ctx_feat = torch.cat([img, cap], dim=0)            # (L_ctx,B,D)
        ctx_mask = torch.cat([img_mask, cap_mask], dim=1)  # (B,L_ctx)

        ctx_feat_cap = torch.cat([img, scan.clone()], dim=0)            # (L_ctx_cap,B,D)
        ctx_mask_cap = torch.cat([img_mask, scan_mask], dim=1)  # (B,L_ctx_cap)


        # ---------- 6) ENCODER: refine scan_feat with context ----------
        for i in range(len(self.enc_scan_self)):
            scan = self.enc_scan_self[i](tgt=scan, tgt_key_padding_mask=scan_mask)[0]
            scan = self.enc_scan_cross[i](tgt=scan, memory=ctx_feat, memory_key_padding_mask=ctx_mask)[0]
            scan = scan.masked_fill(scan_mask.T.unsqueeze(-1), 0.0)
            scan = self.enc_scan_ffn[i](scan)
            scan = scan.masked_fill(scan_mask.T.unsqueeze(-1), 0.0)
        # refined scan: (T,B,D)

        # tag
        for i in range(len(self.enc_scan_self)):
            cap = self.enc_cap_self[i](tgt=cap, tgt_key_padding_mask=cap_mask)[0]
            cap = self.enc_cap_cross[i](tgt=cap, memory=ctx_feat_cap, memory_key_padding_mask=ctx_mask_cap)[0]
            cap = cap.masked_fill(cap_mask.T.unsqueeze(-1), 0.0)
            cap = self.enc_cap_ffn[i](cap)
            cap = cap.masked_fill(cap_mask.T.unsqueeze(-1), 0.0)
        
        refine_scan = torch.cat([scan, cap], dim=0)
        refine_scan_mask = torch.cat([scan_mask, cap_mask], dim=1)


        # ---------- 7) LATENTS ----------
        # z_subj via multi-token query attending to refined scan sequence
        subj_query = self.subj_query_tokens.weight.unsqueeze(1).expand(-1, scan.size(1), -1)  # (Nq,B,D)
        subj_mem, _ = self.pool_subj_cross(tgt=subj_query, memory=refine_scan, memory_key_padding_mask=refine_scan_mask)  # (Nq,B,D)
        # attention-weighted pooling across Nq tokens (learned via tanh -> softmax)
        att_logits = torch.tanh(subj_mem)                  # (Nq,B,D)
        att_scores = F.softmax(att_logits.mean(dim=-1), dim=0)  # (Nq,B)
        z_subj = (att_scores.unsqueeze(-1) * subj_mem).sum(dim=0)  # (B,D)
        z_subj = self.z_subj_mlp(z_subj)                       # (B,D)

        # z_scan via single query token

        scan_query = self.scan_query_token.expand(-1, scan.size(1), -1)  # (1,B,D)
        scan_mem, _ = self.pool_scan_cross(tgt=scan_query, memory=refine_scan, memory_key_padding_mask=refine_scan_mask)  # (1,B,D)
        z_scan = self.z_scan_mlp(scan_mem.squeeze(0))  # (B,D)


        # ---------- 8) Subject classification from z_subj ----------
        subj_logits = self.classifier(z_subj)  # (B,num_subjects)

        # ---------- 9) PARALLEL DECODER (no teacher forcing) ----------
        recon: Dict[str, torch.Tensor] = {}
        losses: Dict[str, torch.Tensor] = {}

        if reconstruct:
            fused = z_scan
            queries = self._build_parallel_queries(B, T, fused)              # (T,B,D)
            dec_h = self._decoder_parallel(z_subj, queries, ctx_feat, ctx_mask, fused)
            pack = self._outputs_and_loss(dec_h, bboxes, scan_mask)
            recon.update(pack["recon"])
            losses.update(pack["losses"])
            

        return {
            "z_subj": z_subj,                 # (B,D)
            "z_scan": z_scan,                 # (B,D)
            "subj_logits": subj_logits,       # (B,num_subjects)
            "recon": recon,                   # {"recon_scan": (B,T,3)}
            "losses": losses                  # {"coord_mse","v_bce","total"} when reconstruct=True
        }



class ScanQwenModel_AE(nn.Module):
    def __init__(self, model, processor, device, hparams):
        super().__init__()
        self.qwen = model
        self.processor = processor 
        self.device = device
        self.hparams = hparams

        # Load ScanAutoEncoder as subject embedding extractor
        self.subj_net = ScanAutoEncoder(model, processor, device, hparams)
        # Projection from AE latent to Qwen hidden dim
        self.subj_to_qwen_proj = nn.Linear(hparams.sen_hidden_size, model.config.hidden_size, bias=False)

    def forward(self, subj_batch, qwen_batch, subj_emb_map=None):
        # Run ScanAutoEncoder (with reconstruction)
        ae_out = self.subj_net(
            images=subj_batch["images"],
            captions=subj_batch["captions"],
            scanpaths=subj_batch["scanpaths"],
            reconstruct=True  # tag
        )
        if self.hparams.stage == 1:
            return {}, ae_out["subj_logits"], ae_out
        
        subj_emb = ae_out["z_subj"]                      # (B, D_sen)
        subj_logits = ae_out["subj_logits"]              # subject classification head

        if subj_emb_map is not None:
            subject_tokens = qwen_batch["subject_names"]
            subj_emb = torch.stack([subj_emb_map[name] for name in subject_tokens], dim=0).to(self.device)
        subj_emb = self.subj_to_qwen_proj(subj_emb)                   # (B, D_qwen)

        # ---- Qwen forward ----
        texts = [self.processor.apply_chat_template(m, tokenize=False) for m in qwen_batch["messages"]]
        image_inputs = [process_vision_info(example)[0] for example in qwen_batch["messages"]]
        model_inputs = self.processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)
        model_inputs = {k: v.to(self.device) for k, v in model_inputs.items()}
        labels = model_inputs["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        # mask image tokens in labels
        image_tokens = [151652, 151653, 151655]
        for tid in image_tokens:
            labels[labels == tid] = -100
        embed_layer = self.qwen.get_input_embeddings()
        orig_forward = embed_layer.forward
        subject_tokens = qwen_batch["subject_names"]

        def new_forward(input_ids: torch.Tensor):
            weight = embed_layer.weight.clone()
            token_ids = self.processor.tokenizer.convert_tokens_to_ids(subject_tokens)
            token_ids = torch.tensor(token_ids, device=weight.device)
            weight[token_ids] = subj_emb.to(weight.dtype)
            return F.embedding(input_ids, weight)


        if self.hparams.stage != 4:
            embed_layer.forward = new_forward
        outputs = self.qwen(**model_inputs, labels=labels)
        embed_layer.forward = orig_forward
        # outputs = {}

        return outputs, subj_logits, ae_out
    

    @torch.no_grad()
    def inference(self, val_batch, processor, device, subj_emb_map, max_new_tokens=512):
        """
        val_batch: one batch from val_loader (batch_size=1)
        subject_tokens: list[str], e.g. ["aanw_84", "dnus_22", ...]
        subject_embs: iterable[torch.Tensor], precomputed subject embeddings
        processor: Qwen2VLProcessor
        device: torch.device
        subj_emb_map: optional dict {subject_name: subject_embedding}, if subj_emb_map is given, 
        subject_tokens and subject_embs are ignored, the model will load subject embedding from pretrained weights
        rather than computing subject embedding in eval time
        Returns: dict {predicted_caption, predicted_subject_id, true_subject_id, ...}
        """
        self.eval()
        qwen_batch = val_batch["qwen"]
        subj_batch = val_batch["subj"]

        # ---- 1. Extract subject embedding from ScanAutoEncoder ----
        ae_out = self.subj_net(
            images=subj_batch["images"],
            captions=subj_batch["captions"],
            scanpaths=subj_batch["scanpaths"],
            reconstruct=True,
        )
        if self.hparams.stage == 1:
            return {}, ae_out
        
        subj_logits = ae_out["subj_logits"]
        pred_ids = subj_logits.argmax(dim=-1)

        subject_tokens = qwen_batch["subject_names"]
        subject_embs = torch.stack([subj_emb_map[name] for name in subject_tokens], dim=0).to(self.device)
        subject_embs = self.subj_to_qwen_proj(subject_embs)                   # (B, D_qwen)
        subject_embs = subject_embs.to(dtype=self.qwen.get_input_embeddings().weight.dtype)
    
        # ---- 2. Patch Qwen embedding table with these subject embeddings ----
        embed_layer = self.qwen.get_input_embeddings()
        orig_forward = embed_layer.forward  # backup
        token_ids = processor.tokenizer.convert_tokens_to_ids(subject_tokens)

        # insert subject embeddings into the corresponding user token ids in the embedding table
        def new_forward(input_ids: torch.Tensor):
            weight = embed_layer.weight.clone()
            token_ids_t = torch.tensor(token_ids, device=weight.device)
            weight[token_ids_t] = subject_embs.to(weight.dtype)
            return F.embedding(input_ids, weight)


        embed_layer.forward = new_forward

        # ---- 3. Prepare message for generation ----
        messages = [_strip_assistant_turns(msg) for msg in qwen_batch["messages"]]
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,  # <-- crucial fix
        ).to(device)

        # ---- 4. Generate caption ----
        output_ids = self.qwen.generate(**inputs, max_new_tokens=max_new_tokens)

        # Slice new tokens after prompt
        gen_ids = [out[len(inp):] for inp, out in zip(inputs.input_ids, output_ids)]
        pred_caption = processor.batch_decode(gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)


        results = save_prediction(pred_caption, qwen_batch["messages"])

        for i in range(subject_embs.size(0)):
            results[i]["predicted_subject_id"] = pred_ids[i].detach().cpu().item()
            results[i]["true_subject_id"] = subj_batch["subject_ids"][i].item()
        # restore embedding forward
        embed_layer.forward = orig_forward

        return results, ae_out