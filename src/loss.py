"""
loss.py — contrastive training utilities (cross‑GPU gather + subject memory bank)

Usage (in train.py):
    from loss import (
        supervised_contrastive_loss,
        ddp_all_gather_no_grad,
        SubjectMemoryBank,
    )
"""

from __future__ import annotations

from typing import Tuple, Optional, List
import torch
import torch.nn.functional as F
import torch.distributed as dist


__all__ = [
    "supervised_contrastive_loss",
    "ddp_all_gather_no_grad",
    "SubjectMemoryBank",
]


# ---------- Supervised Contrastive Loss against a bank ----------
def supervised_contrastive_loss(
    anchors: torch.Tensor,
    anchor_labels: torch.Tensor,
    bank: torch.Tensor,
    bank_labels: torch.Tensor,
    temperature: float = 0.07,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute supervised contrastive loss using a (possibly large) memory bank.

    Args:
        anchors: (A, D) float tensor. Gradients flow through these.
        anchor_labels: (A,) long tensor with subject IDs.
        bank: (N, D) float tensor (typically detached); can be empty (N=0).
        bank_labels: (N,) long tensor with subject IDs for each bank vector.
        temperature: softmax temperature.
        eps: numerical stability epsilon.

    Returns:
        Scalar tensor (loss). Returns 0.0 if no valid positives exist.
    """
    if anchors.numel() == 0:
        return anchors.new_tensor(0.0)
    if bank.numel() == 0:
        return anchors.new_tensor(0.0)

    # Normalize
    z_a = F.normalize(anchors, dim=1)
    z_b = F.normalize(bank, dim=1)

    # Similarities (A, N)
    sim = (z_a @ z_b.T) / temperature

    # Positives: same subject id
    pos = (anchor_labels.view(-1, 1) == bank_labels.view(1, -1))  # (A, N)

    # Log-softmax over bank dimension
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)  # (A, N)

    pos_float = pos.float()
    pos_cnt = pos_float.sum(dim=1)  # (A,)
    valid = pos_cnt > 0

    if not torch.any(valid):
        return anchors.new_tensor(0.0)

    loss_vec = torch.zeros_like(pos_cnt, dtype=sim.dtype)
    loss_vec[valid] = -(log_prob[valid] * pos_float[valid]).sum(dim=1) / (pos_cnt[valid] + eps)

    return loss_vec[valid].mean()


# ---------- Cross-GPU gather that supports variable batch sizes ----------
def ddp_all_gather_no_grad(x: torch.Tensor) -> torch.Tensor:
    """
    All-gather a tensor across ranks (detached) and return the concatenated result.
    Supports variable batch sizes across ranks by padding and trimming.

    If torch.distributed is not initialized, returns x.detach().
    """
    if not (dist.is_available() and dist.is_initialized()):
        return x.detach()

    world_size = dist.get_world_size()
    if world_size == 1:
        return x.detach()

    device = x.device
    b = torch.tensor([x.size(0)], device=device, dtype=torch.long)
    b_list = [torch.zeros_like(b) for _ in range(world_size)]
    dist.all_gather(b_list, b)

    sizes = [int(t.item()) for t in b_list]
    max_b = max(sizes)

    if max_b == 0:
        # All empty
        return x.new_zeros((0,) + x.shape[1:]).detach()

    # Pad current rank to max_b along dim 0
    if x.size(0) < max_b:
        pad_shape = (max_b - x.size(0),) + x.shape[1:]
        pad = torch.zeros(pad_shape, dtype=x.dtype, device=device)
        x_pad = torch.cat([x, pad], dim=0)
    else:
        x_pad = x

    gather_list = [torch.zeros_like(x_pad) for _ in range(world_size)]
    dist.all_gather(gather_list, x_pad)

    # Trim per-rank contributions back to their true sizes
    chunks: List[torch.Tensor] = []
    for tens, sz in zip(gather_list, sizes):
        if sz > 0:
            chunks.append(tens[:sz])
    if len(chunks) == 0:
        return x.new_zeros((0,) + x.shape[1:]).detach()
    return torch.cat(chunks, dim=0).detach()


# ---------- Subject Memory Bank (1 representative per subject + FIFO negatives) ----------
class SubjectMemoryBank:
    """
    Keeps at most one latest representative vector per subject_id and a FIFO
    queue of past embeddings as additional negatives.
    All stored tensors are detached clones on the given device.
    """

    def __init__(
        self,
        emb_dim: int,
        max_negatives: int = 4096,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        self.emb_dim = int(emb_dim)
        self.max_neg = int(max_negatives)
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.subj_to_vec: dict[int, torch.Tensor] = {}
        self.neg_queue: List[Tuple[torch.Tensor, int]] = []

    @torch.no_grad()
    def update(self, emb: torch.Tensor, labels: torch.Tensor) -> None:
        """
        Update the bank with a mini-batch.
        Args:
            emb: (B, D) float tensor
            labels: (B,) long tensor
        """
        if emb.numel() == 0:
            return
        emb = emb.detach()
        labels = labels.detach().long()
        if self.dtype is not None and emb.dtype != self.dtype:
            emb = emb.to(self.dtype)
        emb = emb.to(self.device)
        labels = labels.to(self.device)

        # One representative per subject (latest wins)
        for v, l in zip(emb, labels):
            self.subj_to_vec[int(l)] = v

        # Add all to FIFO negatives
        for v, l in zip(emb, labels):
            self.neg_queue.append((v, int(l)))
        if len(self.neg_queue) > self.max_neg:
            self.neg_queue = self.neg_queue[-self.max_neg:]

    @torch.no_grad()
    def build_bank(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build the (vectors, labels) tensors comprising:
          * one representative per subject_id
          * plus the FIFO negative queue
        Returns empty tensors if nothing stored yet.
        """
        vecs: List[torch.Tensor] = []
        labs: List[int] = []

        for lid, vec in self.subj_to_vec.items():
            vecs.append(vec)
            labs.append(lid)

        if self.neg_queue:
            q_vecs, q_labs = zip(*self.neg_queue)
            vecs.extend(q_vecs)
            labs.extend(q_labs)

        if not vecs:
            return (
                torch.empty(0, self.emb_dim, device=self.device, dtype=self.dtype or torch.float32),
                torch.empty(0, dtype=torch.long, device=self.device),
            )
        return torch.stack(vecs, dim=0), torch.tensor(labs, device=self.device, dtype=torch.long)
