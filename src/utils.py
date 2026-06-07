import os
import sys
import copy
import json
import torch
import random

import numpy as np
import torch.distributed as dist
import torch.nn.functional as F

from safetensors.torch import load_file, save_file
from peft import PeftModel, get_peft_model_state_dict
from tqdm import tqdm
from typing import List, Dict, Any
from collections import defaultdict, Counter
from pathlib import Path
from peft import LoraConfig, get_peft_model, PeftModel
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from loss import supervised_contrastive_loss, ddp_all_gather_no_grad, SubjectMemoryBank

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'cococaption')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'pycocoevalcap')))

from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap


def print_rank0(*args, **kwargs):
    """Print only from rank 0 process."""
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(*args, **kwargs)

def get_special_token_list_coco():
    return [
        "<ekbs_0>", "<fqmz_1>", "<kecc_2>", "<ymhv_3>", "<jkyr_4>", "<kcwu_5>", 
        "<ivtj_6>", "<sumc_7>", "<thte_8>", "<ydqf_9>", "<xzjy_12>", "<kxdo_13>", 
        "<vuug_14>", "<gxuy_15>", "<cfpq_17>", "<wkys_18>", "<cgwe_19>", "<abyw_20>", 
        "<unrf_21>", "<dnus_22>", "<arel_24>", "<jlzd_25>", "<saiv_26>", "<eixu_27>", 
        "<krpm_28>", "<msfc_31>", "<qhcn_32>", "<gmhy_33>", "<alvk_34>", "<pnir_35>", 
        "<gfib_36>", "<pigc_37>", "<sxwp_38>", "<uvfk_39>", "<lewi_40>", "<fioj_41>", 
        "<itza_42>", "<tbug_43>", "<xwsn_44>", "<xlvd_45>", "<gjdq_46>", "<orsy_47>", 
        "<glpf_48>", "<ugti_49>", "<xcgl_50>", "<fqqb_51>", "<eynb_52>", "<nnlf_53>", 
        "<mrwx_55>", "<sjzw_56>", "<fgkh_58>", "<gavy_59>", "<qcoa_60>", "<fwpb_61>", 
        "<vufb_62>", "<yjnk_64>", "<llaz_66>", "<jirj_67>", "<wazq_68>", "<pdpd_69>", 
        "<avpv_70>", "<tufh_71>", "<uips_72>", "<yekj_73>", "<tjnp_74>", "<igez_75>", 
        "<bojz_76>", "<ssti_77>", "<tkqf_78>", "<hjzi_79>", "<lcmj_80>", "<aqxw_81>", 
        "<rmne_82>", "<zmrv_83>", "<aanw_84>", "<kgox_85>", "<lbqp_87>", "<vqta_88>", 
        "<giri_89>", "<ckgh_90>", "<ykey_91>", "<lisc_92>", "<rdzu_93>", "<egcw_94>", 
        "<irqv_95>", "<gvsu_96>", "<lmhk_97>", "<kslb_99>", "<gwlf_100>", "<sebg_102>", 
        "<zonp_103>", "<phyc_105>", "<knkx_107>", "<aczw_110>", "<abmj_111>", "<ltqn_113>", 
        "<iubd_114>", "<fdpm_115>", "<ghar_116>", "<edtz_117>", "<abeu_118>", "<jsxz_119>", 
        "<dldj_120>", "<bfdf_121>", "<tkos_122>", "<mdxu_123>", "<kslr_124>", "<qquc_125>"
    ]

def get_special_token_list_flickr30k():
    return [
        '<vnzl_0>', '<nexy_1>', '<wgom_2>', '<lord_3>', '<svgq_4>', '<oxks_5>', '<anhi_6>', 
        '<naug_7>', '<eflr_8>', '<bzsr_9>', '<zfhi_10>', '<ougw_11>', '<asgu_12>', '<vcrg_13>', 
        '<otsf_14>', '<itjq_15>', '<qgmz_16>', '<mnrf_17>', '<hzam_18>', '<gdan_19>', 
        '<xtnt_20>', '<gidl_21>', '<dsbr_22>', '<nlhg_23>', '<ibay_24>', '<dwfd_25>', 
        '<mtno_26>', '<jtch_27>', '<kpwg_28>', '<nhzp_29>', '<vllp_30>', '<hhik_31>', 
        '<qzvt_32>', '<ciyo_33>', '<zpam_34>', '<bjsg_35>', '<hyrx_36>']

def get_special_token_list_SenHe():
    return ['<ztnc_0>', '<qvnf_1>', '<jeyx_2>', '<xpmc_3>', '<jzug_4>']

def get_special_token_list_DKollenda():
    return ['<fqof_0>', '<pvau_1>', '<siey_2>', '<iccw_3>', '<pusn_4>', '<zjov_5>', '<qwps_6>', '<bfhc_7>', '<gchq_8>', 
            '<jjfg_9>', '<yqpe_10>', '<sejz_11>', '<qorv_12>', '<ufai_13>', '<gfyw_14>', '<irkx_15>', '<lggo_16>', 
            '<gpxk_17>', '<fznc_18>', '<bcqu_19>', '<kbjz_20>', '<nzwa_21>', '<srng_22>', '<qcll_23>', '<ywgn_24>', 
            '<exwh_25>', '<qpdt_26>', '<ouna_27>', '<iayw_28>', '<vhbw_29>']

def get_special_token_list_population():
    return ['<sks>']

    
def init_peft(base_model, old_vocab_size, args):
    if args.stage == 4:
        peft_config = LoraConfig(
            lora_alpha=16, lora_dropout=0.05, r=8, bias="none",
            target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM",
            modules_to_save=["embed_tokens"]
        )
    else:
        peft_config = LoraConfig(
            lora_alpha=16, lora_dropout=0.05, r=8, bias="none",
            target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM",
            # modules_to_save=["embed_tokens", "lm_head"] 
        ) 
    peft_model = get_peft_model(base_model, peft_config)
    def freeze_old_rows(grad):
        grad[:old_vocab_size] = 0  # zero out old token grads
        return grad

    # Attach hook to embeddings
    if args.stage == 4:
        embed_tokens = peft_model.get_input_embeddings()
        embed_tokens.weight.register_hook(freeze_old_rows)

    return peft_model


def load_checkpoint(base_model, finetune_ckp):
    """
    Load a Qwen2 model with LoRA adapters.
    base_model: the already loaded Qwen2VLForConditionalGeneration
    finetune_ckp: path to folder containing adapter_model.safetensors
    """
    # --- Fix potential bad key names in adapter file ---
    in_path  = f"{finetune_ckp}/adapter_model.safetensors"
    out_path = f"{finetune_ckp}/adapter_model.safetensors"

    weights = load_file(in_path)
    new_weights = {}
    for k, v in weights.items():
        new_k = copy.deepcopy(k)
        # collapse duplicate prefixes if present
        if "base_model.model.base_model.model.model" in new_k:
            new_k = new_k.replace(
                "base_model.model.base_model.model.model.",
                "base_model.model.model.", 1
            )
        new_weights[new_k] = v

    # overwrite the safetensors file with cleaned keys
    save_file(new_weights, out_path)

    # --- Load LoRA adapter on top of base model ---
    model = PeftModel.from_pretrained(
        base_model,
        finetune_ckp,
        is_trainable=False  # inference only
    )
    model.set_adapter("default")

    print(f"✅ Loaded LoRA finetune from {finetune_ckp}")
    return model


def load_checkpoint_new(model, processor, special_tokens, ckp_path):
    print(f'finetune subject from {ckp_path}')
    # load adapter for language model
    in_path  = f"{ckp_path}/adapter_model.safetensors"
    out_path = f"{ckp_path}/adapter_model.safetensors"

    weights = load_file(in_path)
    new_weights = {}
    for k, v in weights.items():
        new_k = copy.deepcopy(k)
        # 1. collapse duplicate "base_model.model.base_model.model." → "base_model.model.model."
        if "base_model.model.base_model.model.model" in new_k:
            new_k = new_k.replace("base_model.model.base_model.model.model.", "base_model.model.model.", 1)

        new_weights[new_k] = v

    # save cleaned file
    # save_file(new_weights, out_path)
    model = PeftModel.from_pretrained(model, f"{ckp_path}", is_trainable=False)
    model.set_adapter("default")

    # # load token and lm_head
    state = load_file(f"{ckp_path}/personality_adapter.safetensors")
    embed_layer = model.get_input_embeddings()
    lm_head = model.lm_head

    vocab_dim0, vocab_dim1 = lm_head.weight.shape
    vocab_size = processor.tokenizer.vocab_size

    for tok in special_tokens:
        tok_id = processor.tokenizer.convert_tokens_to_ids(tok)
        if tok_id == processor.tokenizer.unk_token_id:
            continue

        key = f"embed_tokens.{tok_id}"
        if key in state:
            embed_layer.weight.data[tok_id] = state[key].to(embed_layer.weight.device)

        key = f"lm_head.{tok_id}"
        if key in state:
            lm_head.weight.data[tok_id] = state[key].to(lm_head.weight.device)

    for k, v in model.named_parameters():
        if 'lora' in k and 'lm_head' not in k:
            v.requires_grad = False
    print(f"✅ Restored {len(special_tokens)} special tokens into model.")
    return model
    

# Strip out assistant (gt caption) turns to avoid leaking GT
def _strip_assistant_turns(messages):
    """Keep only system + user turns to avoid leaking GT."""
    kept = []
    for m in messages:
        role = (m.get("role") or "").lower()
        if role in ("user"):
            kept.append(m)
    return kept


# def save_prediction(output_text, val_message):
#     pred = output_text[0]
#     img_path = val_message[1]['content'][0]['image']
#     reference = val_message[2]['content'][0]['text']
#     subject = val_message[1]['content'][2]['name']
#     img_id = os.path.basename(img_path).split('.')[0]
#     record = {"id": img_id, "prediction": pred, 'reference': reference, 'subject': subject}
#     return record


def save_prediction(pred_texts, qwen_messages_batch):
    """
    Batch version of save_prediction that supports multiple samples.

    Args:
        pred_texts (list[str]): model-generated captions (len = batch size)
        qwen_messages_batch (list[list[dict]]): messages per sample, from collate_fn.

    Returns:
        list[dict]: each entry is a result dictionary ready for JSON writing.
    """
    records = []
    for pred, messages in zip(pred_texts, qwen_messages_batch):
        try:
            img_path = messages[1]['content'][3]['path']
            reference = messages[2]['content'][0]['text']
            subject = messages[1]['content'][2]['name']
        except Exception:
            img_path, reference, subject = None, None, None

        img_id = os.path.basename(img_path).split('.')[0] if img_path else "unknown"
        records.append({
            "id": img_id,
            "prediction": pred,
            "reference": reference,
            "subject": subject
        })
    return records


def compute_caption_score(out_path):
    in_path = Path(out_path)
    gt_path = 'gt_coco_format.json'
    pred_path = 'prediction_coco_format.json'

    def to_int_id(s):
        try:
            return int(str(s))
        except Exception:
            return int("".join([c for c in str(s) if c.isdigit()]) or 0)

    with in_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Aggregate by image_id
    refs_by_img = defaultdict(list)
    preds_by_img_all = defaultdict(list)

    for idx, item in enumerate(data):
        img_str_id = item.get("id")
        img_id = to_int_id(img_str_id)
        ref = (item.get("reference") or "").strip()
        pred = (item.get("prediction") or "").strip()
        if ref:
            refs_by_img[img_id].append(ref)
        if pred:
            preds_by_img_all[img_id].append({"pred": pred, "pos": idx})

    # Deduplicate references per image (exact match)
    for k, v in list(refs_by_img.items()):
        # preserve order while removing dups
        seen = set()
        deduped = []
        for s in v:
            if s not in seen:
                seen.add(s)
                deduped.append(s)
        refs_by_img[k] = deduped


    preds_by_img = {}
    for img_id, lst in preds_by_img_all.items():
        preds_by_img[img_id] = lst[-1]["pred"]

    # Build COCO GT
    images = [{"id": img_id} for img_id in sorted(refs_by_img.keys())]
    annotations = []
    ann_id = 0
    for img_id, refs in refs_by_img.items():
        for r in refs:
            annotations.append({
                "image_id": img_id,
                "id": ann_id,
                "caption": r
            })
            ann_id += 1

    coco_gt = {
        "info": {},
        "licenses": [],
        "type": "captions",
        "images": images,
        "annotations": annotations
    }

    # Build COCO predictions
    coco_pred = [{"image_id": img_id, "caption": cap}
                for img_id, cap in sorted(preds_by_img.items())]

    with open(gt_path, 'w', encoding="utf-8") as f:
        json.dump(coco_gt, f, ensure_ascii=False, indent=2)

    with open(pred_path, 'w', encoding="utf-8") as f:
        json.dump(coco_pred, f, ensure_ascii=False, indent=2)


    coco = COCO(gt_path)
    cocoRes = coco.loadRes(pred_path)
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.params['image_id'] = cocoRes.getImgIds()
    cocoEval.evaluate()

    return cocoEval.eval


def compute_emb_score(y_true, y_pred, average="macro"):
    """
    Compute Precision, Recall, and F1 for subject ID predictions.

    Args:
        y_true (list or np.ndarray): Ground-truth subject IDs
        y_pred (list or np.ndarray): Predicted subject IDs
        average (str): Averaging type for multi-class ['micro', 'macro', 'weighted']

    Returns:
        dict: {"precision": float, "recall": float, "f1": float}
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    precision = precision_score(y_true, y_pred, average=average, zero_division=0)
    recall    = recall_score(y_true, y_pred, average=average, zero_division=0)
    f1        = f1_score(y_true, y_pred, average=average, zero_division=0)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1)
    }


def get_trainable_state_dict(model: torch.nn.Module):
    trainable = {}
    param_dict = dict(model.named_parameters())
    for k, v in model.state_dict().items():
        if k in param_dict and param_dict[k].requires_grad:
            trainable[k] = v
    return trainable


def is_main():
    return (not dist.is_initialized()) or dist.get_rank() == 0


def print_rank0(*args, **kwargs):
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(*args, **kwargs)

def setup_seed(seed: int):
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_ddp():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    return local_rank

def is_main():
    return (not dist.is_initialized()) or dist.get_rank() == 0



def train_senet_one_epoch(model, loader, optimizer, device, epoch, args, global_step, log_file_loss,
                    val_loader, processor, local_rank, bank=None):
    model.train()
    if isinstance(loader.sampler, DistributedSampler):
        loader.sampler.set_epoch(epoch)

    pbar = tqdm(loader, disable=not is_main(), dynamic_ncols=True)
    accum = args.grad_accum_steps

    for step, batch in enumerate(pbar, start=1):
        global_step += 1
        subj_batch = {
            k: (v.to(local_rank, non_blocking=True) if isinstance(v, torch.Tensor) else v)
            for k, v in batch["subj"].items()
        }
        qwen_batch = {
            k: (v.to(local_rank, non_blocking=True) if isinstance(v, torch.Tensor) else v)
            for k, v in batch["qwen"].items()
        }
        outputs, subj_logits, ae_out = model(subj_batch, qwen_batch)
        # AE loss (from model)
        losses = ae_out["losses"]
        ae_loss = losses["total"] if "total" in losses else torch.tensor(0.0, device=device)
        iou_loss = losses["iou_mean"] if "iou_mean" in losses else torch.tensor(0.0, device=device)
        cls_loss = torch.tensor(0.0, device=device)
        cls_logits = ae_out["subj_logits"]  # (B, num_subjects)
        cls_loss = F.cross_entropy(cls_logits, subj_batch["subject_ids"])

        # contrastive loss
        local_emb  = ae_out['z_subj']
        local_lbls = subj_batch["subject_ids"].to(local_emb.device).long()
        global_emb = ddp_all_gather_no_grad(local_emb)
        global_lbl = ddp_all_gather_no_grad(local_lbls)
        bank.update(global_emb, global_lbl)
        bank_vecs, bank_lbls = bank.build_bank()

        contrast_loss = supervised_contrastive_loss(
            anchors=local_emb,
            anchor_labels=local_lbls,
            bank=bank_vecs,
            bank_labels=bank_lbls,
            temperature=args.contrast_temp,
        )

        # total_loss = ae_loss + 30 * iou_loss + args.lambda_subj * cls_loss + args.lambda_contrast * contrast_loss # tag
        total_loss = ae_loss + args.lambda_subj * cls_loss + args.lambda_contrast * contrast_loss
        # total_loss = args.lambda_subj * cls_loss + args.lambda_contrast * contrast_loss
        # total_loss = ae_loss + 30 * iou_loss + args.lambda_subj * cls_loss
        # ---------------- backward ----------------
        total_loss = total_loss / accum
        total_loss.backward()

        if (step % accum) == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            optimizer.zero_grad()

        # ---------------- logging ----------------
        # reduce across ranks for clean logs
        tot = total_loss.detach().clone() * accum
        cm  = losses["bbox_l1"].detach().clone() if "bbox_l1" in losses else torch.tensor(0.0, device=device)
        iou = losses["iou_mean"].detach().clone() if "iou_mean" in losses else torch.tensor(0.0, device=device)
        vb  = losses["v_bce"].detach().clone() if "v_bce" in losses else torch.tensor(0.0, device=device)
        cl  = cls_loss.detach().clone()
        contrastive = contrast_loss.detach().clone()

        if dist.is_initialized():
            for t in (tot, cm, vb, cl, contrastive):
                dist.all_reduce(t, op=dist.ReduceOp.AVG)

        if global_step % args.print_every == 0 and is_main():
            rec = {
                "epoch": epoch,
                "global_step": global_step,
                "loss/total": float(tot.item()),
                "loss/bbox_l1": float(cm.item()),
                "loss/iou_mean": float(iou.item()),
                "loss/v_bce": float(vb.item()),
                "loss/subj_cls": float(cl.item()),
                "loss/contrastive": float(contrastive.item()),
                "lr": optimizer.param_groups[0]["lr"],
            }
            with open(log_file_loss, "a") as f:
                f.write(json.dumps(rec) + "\n")
            pbar.set_description(
                f"[E{epoch} | GS {global_step}] "
                f"tot {rec['loss/total']:.4f} | bbox_l1 {rec['loss/bbox_l1']:.4f} | iou_mean {rec['loss/iou_mean']:.4f} | v_bce {rec['loss/v_bce']:.4f} | cls {rec['loss/subj_cls']:.4f} | contrastive {rec['loss/contrastive']:.4f}"
            )

        # ---------------- save ----------------
        if (global_step % args.save_every) == 0 and is_main() and (epoch >=30 or args.eval_every==1):  # tag
            save_state(model, optimizer, epoch, global_step, args)

        # ---------------- EVAL MID-EPOCH ----------------
        # Evaluate exactly when global_step % eval_every == 0 (no need to complete an epoch)
        if args.eval_every > 0 and global_step > 0 and (global_step % args.eval_every == 0) and (epoch >=20 or args.eval_every==1):
            # run eval
            metrics = eval_senet_greedy(
                model=model,
                loader=val_loader,
                device=device,
                epoch=epoch,                     # keep epoch context
                global_step=global_step,
                args=args,
                processor=processor,
                local_rank=local_rank,
                max_batches=args.eval_batch_num
            )
            if is_main():
                rec = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "eval/bbox_l1": None if metrics["bbox_l1"] is None else float(metrics["bbox_l1"]),
                    "eval/iou_mean": None if metrics["iou_mean"] is None else float(metrics["iou_mean"]),
                    "eval/v_bce": None if metrics["v_bce"] is None else float(metrics["v_bce"]),
                    "eval/ae_total": None if metrics["ae_total"] is None else float(metrics["ae_total"]),
                    "eval/cls_ce": None if metrics["cls_ce"] is None else float(metrics["cls_ce"]),
                    "eval/cls_acc": None if metrics["cls_acc"] is None else float(metrics["cls_acc"]),
                    "eval/total_with_cls": None if metrics["total_with_cls"] is None else float(metrics["total_with_cls"]),
                }
                with open(os.path.join(args.save_ckp_path, "eval_log.jsonl"), "a") as f:
                    f.write(json.dumps(rec) + "\n")
                print_rank0(f"[Eval@GS{global_step}] {rec}")
            # go back to train mode
            model.train()

    return global_step


@torch.no_grad()
def eval_senet_greedy(model, loader, device, epoch, global_step, args, processor, local_rank, max_batches=20):
    """
    Writes predictions to:
      <save_ckp_path>/eval_preds_epoch{epoch}.jsonl

    Each line:
    {
        "image_name": str | null,
        "subject": int | null,          # kept for backward compatibility
        "subject_gt": int | null,       # ground-truth subject id
        "subject_pred": int | null,     # predicted subject id (argmax)
        "subject_pred_conf": float | null,  # confidence of predicted class (max softmax prob)
        "caption": str | null,
        "scanpath": [[x,y,v], ...]      # predicted, floats in [0,1]
    }

    Returns averaged metrics (DDP-reduced):
      - coord_mse, v_bce, ae_total
      - cls_ce (if GT available), cls_acc (if GT available)
      - total_with_cls = ae_total + lambda_subj * cls_ce
    """
    model.eval()
    pbar = tqdm(loader, disable=not is_main(), desc=f"[Eval@E{epoch}]", dynamic_ncols=True)

    # running means
    n_seen = 0
    bbox_l1_sum = 0.0
    iou_mean_sum = 0.0
    v_bce_sum = 0.0
    ae_total_sum = 0.0
    cls_ce_sum = 0.0
    cls_acc_sum = 0.0
    n_seen_cls = 0  # only count batches with GT subject_ids

    preds_path = os.path.join(args.save_ckp_path, f"eval_preds_step{global_step}.jsonl")
    preds_f = None
    if is_main():
        preds_f = open(preds_path, "w", encoding="utf-8")

    tokenizer = getattr(processor, "tokenizer", None)

    for i, batch in enumerate(pbar, start=1):
        if i > max_batches:
            break

        subj_batch = {
            k: (v.to(local_rank, non_blocking=True) if isinstance(v, torch.Tensor) else v)
            for k, v in batch["subj"].items()
        }
        qwen_batch = {
            k: (v.to(local_rank, non_blocking=True) if isinstance(v, torch.Tensor) else v)
            for k, v in batch["qwen"].items()
        }
        outputs, subj_logits, ae_out = model(subj_batch, qwen_batch)

        # ---- AE losses
        losses = ae_out["losses"]
        cm = losses["bbox_l1"].detach().clone() if "bbox_l1" in losses else torch.tensor(0.0, device=device)
        iou = losses["iou_mean"].detach().clone() if "iou_mean" in losses else torch.tensor(0.0, device=device)
        vb = losses["v_bce"].detach().clone() if "v_bce" in losses else torch.tensor(0.0, device=device)
        ae_tot = losses["total"].detach().clone() if "total" in losses else torch.tensor(0.0, device=device)

        if dist.is_initialized():
            for t in (cm, iou, vb, ae_tot):
                dist.all_reduce(t, op=dist.ReduceOp.AVG)

        bbox_l1_sum += cm.item()
        iou_mean_sum += iou.item()
        v_bce_sum += vb.item()
        ae_total_sum += ae_tot.item()
        n_seen += 1

        # ---- subject logits / predictions
        logits = ae_out.get("subj_logits", None)  # (B, num_subjects)
        if logits is not None:
            probs = torch.softmax(logits, dim=-1)
            pred_ids = probs.argmax(dim=-1)                # (B,)
            pred_conf = probs.max(dim=-1).values           # (B,)
        else:
            pred_ids = None
            pred_conf = None

        # ---- classification loss/acc if GT available
        subj_ids_tensor = subj_batch["subject_ids"]
        if isinstance(subj_ids_tensor, torch.Tensor) and (logits is not None):
            subj_ids_tensor = subj_ids_tensor.to(device, non_blocking=True)
            ce = F.cross_entropy(logits, subj_ids_tensor).detach().clone()
            acc = (pred_ids == subj_ids_tensor).float().mean().detach().clone()
            if dist.is_initialized():
                dist.all_reduce(ce,  op=dist.ReduceOp.AVG)
                dist.all_reduce(acc, op=dist.ReduceOp.AVG)
            cls_ce_sum  += ce.item()
            cls_acc_sum += acc.item()
            n_seen_cls  += 1

        # ---- progress bar
        if is_main():
            base = f"[Eval@E{epoch}] bbox_l1 {bbox_l1_sum/n_seen:.4f} | iou_mean {iou_mean_sum/n_seen:.4f} | v_bce {v_bce_sum/n_seen:.4f} | ae_total {ae_total_sum/n_seen:.4f}"
            if n_seen_cls > 0:
                base += f" | cls_ce {cls_ce_sum/n_seen_cls:.4f} | cls_acc {cls_acc_sum/n_seen_cls:.4f}"
            pbar.set_description(base)

        # ---- write predictions (main rank only)
        if is_main():
            recon_scan = ae_out["recon"]["recon_scan"].detach().cpu().numpy() if ae_out["recon"] else None  # (B,T,3)
            
            # image names (best-effort)
            img_names = subj_batch.get("image_names") or subj_batch.get("image_name") or subj_batch.get("img_name") \
                        or subj_batch.get("img_names") or subj_batch.get("img_path") or subj_batch.get("paths")
            B = len(img_names)
            if isinstance(img_names, (list, tuple)):
                img_names_list = [str(x) for x in img_names]
            elif isinstance(img_names, str):
                img_names_list = [img_names]
            else:
                img_names_list = [None] * B

            # ground-truth subjects
            subj_ids_cpu = batch["subj"].get("subject_ids", None)
            if isinstance(subj_ids_cpu, torch.Tensor):
                subj_gt_list = subj_ids_cpu.cpu().tolist()
                if not isinstance(subj_gt_list, list):
                    subj_gt_list = [subj_gt_list]
            elif isinstance(subj_ids_cpu, (list, tuple)):
                subj_gt_list = list(subj_ids_cpu)
            else:
                subj_gt_list = [None] * B

            # predicted subjects and confidences
            if pred_ids is not None:
                subj_pred_list = pred_ids.detach().cpu().tolist()
                subj_pred_conf_list = pred_conf.detach().cpu().tolist()
            else:
                subj_pred_list = [None] * B
                subj_pred_conf_list = [None] * B

            # decode captions (best-effort)
            captions_text = [None] * B
            cap_ids = batch["subj"]["captions"].get("input_ids", None)
            if tokenizer is not None and isinstance(cap_ids, torch.Tensor):
                for bi in range(B):
                    ids = cap_ids[bi].tolist()
                    captions_text[bi] = tokenizer.decode(ids, skip_special_tokens=True)

            # write per-sample record
            for bi in range(B):
                record = {
                    "image_name": img_names_list[bi],
                    "subject_gt": subj_gt_list[bi],
                    "subject_pred": subj_pred_list[bi],
                    "subject_pred_conf": subj_pred_conf_list[bi],
                    "caption": captions_text[bi],
                    "scanpath": recon_scan[bi].tolist() if recon_scan is not None else None,         # list of [x,y,v]
                }
                preds_f.write(json.dumps(record, ensure_ascii=False) + "\n")

    if preds_f is not None:
        preds_f.close()
        print_rank0(f"📄 Wrote eval predictions to {preds_path}")

    if n_seen == 0:
        return {
            "bbox_l1": None, "iou_mean": None, "v_bce": None, "ae_total": None,
            "cls_ce": None, "cls_acc": None, "total_with_cls": None
        }

    bbox_l1_mean = bbox_l1_sum / n_seen
    iou_mean_mean = iou_mean_sum / n_seen
    v_bce_mean     = v_bce_sum / n_seen
    ae_total_mean  = ae_total_sum / n_seen
    cls_ce_mean    = (cls_ce_sum / n_seen_cls) if n_seen_cls > 0 else 0.0
    cls_acc_mean   = (cls_acc_sum / n_seen_cls) if n_seen_cls > 0 else 0.0
    total_with_cls = ae_total_mean + args.lambda_subj * cls_ce_mean

    return {
        "bbox_l1": bbox_l1_mean,
        "iou_mean": iou_mean_mean,
        "v_bce": v_bce_mean,
        "ae_total": ae_total_mean,
        "cls_ce": cls_ce_mean,
        "cls_acc": cls_acc_mean,
        "total_with_cls": total_with_cls,
    }



def save_state(model, optimizer, epoch, global_step, args):
    module = model.module if isinstance(model, DDP) else model

    # --- embedding handles ---
    in_emb = module.qwen.get_input_embeddings()
    out_emb = module.qwen.get_output_embeddings()  # may be tied or None

    state = {
        "epoch": epoch,
        "global_step": global_step,
        "args": vars(args),
        "optimizer": optimizer.state_dict(),
        "subj_net": get_trainable_state_dict(module),
        "subj_to_qwen_proj": module.subj_to_qwen_proj.state_dict(),

        # --- explicitly store embeddings ---
        "embed_tokens": in_emb.state_dict(),                 # {"weight": ...}
        # Save lm_head only if it exists and is NOT tied to input embeddings
        "lm_head": (None if (out_emb is None or out_emb.weight is in_emb.weight)
                    else out_emb.state_dict()),
    }
    try:
        state['lora_net'] = get_peft_model_state_dict(module.qwen)
    except Exception as e:
        pass
    ckp = os.path.join(args.save_ckp_path, f"ckpt_step{global_step}.pth")
    torch.save(state, ckp)
    print_rank0(f"💾 Saved checkpoint to {ckp}")


def joint_train_senet_qwen_one_epoch(model, loader, optimizer, device, epoch, args, global_step, log_file_loss,
                    val_loader, sample_loader, processor, local_rank, subject_id_to_name, bank):
    model.train()
    if isinstance(loader.sampler, DistributedSampler):
        loader.sampler.set_epoch(epoch)

    pbar = tqdm(loader, disable=not is_main(), dynamic_ncols=True)
    accum = args.grad_accum_steps

    for step, batch in enumerate(pbar, start=1):
        global_step += 1
        subj_batch = {
            k: (v.to(local_rank, non_blocking=True) if isinstance(v, torch.Tensor) else v)
            for k, v in batch["subj"].items()
        }
        qwen_batch = {
            k: (v.to(local_rank, non_blocking=True) if isinstance(v, torch.Tensor) else v)
            for k, v in batch["qwen"].items()
        }
        outputs, subj_logits, ae_out = model(subj_batch, qwen_batch)

        # AE loss (from model)
        losses = ae_out["losses"]
        ae_loss = losses["total"] if "total" in losses else torch.tensor(0.0, device=device)
        iou_loss = losses["iou_mean"] if "iou_mean" in losses else torch.tensor(0.0, device=device)
        cls_loss = torch.tensor(0.0, device=device)
        cls_logits = ae_out["subj_logits"]  # (B, num_subjects)
        cls_loss = F.cross_entropy(cls_logits, subj_batch["subject_ids"])
        # qwen loss
        caption_loss = outputs.loss
        local_emb  = ae_out['z_subj']
        local_lbls = subj_batch["subject_ids"].to(local_emb.device).long()
        global_emb = ddp_all_gather_no_grad(local_emb)
        global_lbl = ddp_all_gather_no_grad(local_lbls)
        bank.update(global_emb, global_lbl)
        bank_vecs, bank_lbls = bank.build_bank()

        contrast_loss = supervised_contrastive_loss(
            anchors=local_emb,
            anchor_labels=local_lbls,
            bank=bank_vecs,
            bank_labels=bank_lbls,
            temperature=args.contrast_temp,
        )
        if args.stage == 2:
            # total_loss = caption_loss + args.lambda_subj * cls_loss + args.lambda_contrast * contrast_loss + ae_loss + iou_loss  # tag
            total_loss = caption_loss + args.lambda_subj * cls_loss + args.lambda_contrast * contrast_loss
            # total_loss = caption_loss + args.lambda_subj * cls_loss
        elif args.stage == 3:
            total_loss = caption_loss + args.lambda_subj * cls_loss + args.lambda_contrast * contrast_loss  # tag
        elif args.stage == 4:
            total_loss = caption_loss


        # ---------------- backward ----------------
        total_loss = total_loss / accum
        total_loss.backward()

        if (step % accum) == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            optimizer.zero_grad()

        # ---------------- logging ----------------
        # reduce across ranks for clean logs
        tot = total_loss.detach().clone() * accum
        bbox_l1  = losses["bbox_l1"].detach().clone() if "bbox_l1" in losses else torch.tensor(0.0, device=device)
        iou_mean  = losses["iou_mean"].detach().clone() if "iou_mean" in losses else torch.tensor(0.0, device=device)
        vb  = losses["v_bce"].detach().clone() if "v_bce" in losses else torch.tensor(0.0, device=device)
        cl  = cls_loss.detach().clone()
        ql  = caption_loss.detach().clone()
        contrastive = contrast_loss.detach().clone()

        if dist.is_initialized():
            for t in (tot, bbox_l1, iou_mean, vb, cl, ql, contrastive):
                dist.all_reduce(t, op=dist.ReduceOp.AVG)

        if global_step % args.print_every == 0 and is_main():
            rec = {
                "epoch": epoch,
                "global_step": global_step,
                "loss/total": float(tot.item()),
                "loss/caption": float(ql.item()),
                "loss/bbox_l1": float(bbox_l1.item()),
                "loss/iou_mean": float(iou_mean.item()),
                "loss/v_bce": float(vb.item()),
                "loss/subj_cls": float(cl.item()),
                "loss/contrastive": float(contrastive.item()),
                "lr": optimizer.param_groups[0]["lr"],
            }
            with open(log_file_loss, "a") as f:
                f.write(json.dumps(rec) + "\n")
            pbar.set_description(
                f"[E{epoch} | GS {global_step}] "
                f"tot {rec['loss/total']:.4f} | caption {rec['loss/caption']:.4f} | bbox_l1 {rec['loss/bbox_l1']:.4f} | iou_mean {rec['loss/iou_mean']:.4f} | v_bce {rec['loss/v_bce']:.4f} | cls {rec['loss/subj_cls']:.4f} | contrastive {rec['loss/contrastive']:.4f}"
            )

        # ---------------- save ----------------
        if (global_step % args.save_every) == 0 and is_main() and epoch <= 20:
            save_state(model, optimizer, epoch, global_step, args)

        # ---------------- EVAL MID-EPOCH ----------------
        # Evaluate exactly when global_step % eval_every == 0 (no need to complete an epoch)
        # tag
        if args.eval_every > 0 and global_step > 0 and (global_step % args.eval_every == 0) and (epoch >=6 or args.eval_every==1):
            # run eval
            metrics, pred = eval_qwen_senet_greedy(
                model=model,
                val_loader=val_loader,
                sample_loader=sample_loader,
                device=device,
                epoch=epoch,                     # keep epoch context
                global_step=global_step,
                args=args,
                processor=processor,
                local_rank=local_rank,
                max_batches=args.eval_batch_num,
                subject_id_to_name=subject_id_to_name,
                optimizer=optimizer
            )
            if is_main():
                if args.fewshot_subjects:
                    subject_split = "seen" if args.is_seen else "unseen"
                    with open(f'{args.save_ckp_path}/{subject_split}_stage_{args.stage}_sample_{args.num_fewshot_sample}_seed_{args.seed}_fewshot_eval_log.json', 'w') as f:
                        f.write(json.dumps(metrics, indent=2))
                    with open(f'{args.save_ckp_path}/{subject_split}_stage_{args.stage}_sample_{args.num_fewshot_sample}_seed_{args.seed}_fewshot_eval_pred.json', 'w') as f:
                        f.write(json.dumps(pred, indent=2))
                else:
                    with open(os.path.join(args.save_ckp_path, "eval_scan_log.jsonl"), "a") as f:
                        f.write(json.dumps(metrics) + "\n")
            model.train()

    return global_step


@torch.no_grad()
def eval_qwen_senet_greedy(model, val_loader, sample_loader, device, epoch,
                           global_step, args, processor, local_rank, max_batches,
                           subject_id_to_name, optimizer):
    """
    DDP + batch-size compatible version of eval_qwen_senet_greedy.
    Each GPU evaluates its shard; rank 0 aggregates results and metrics.
    """
    model.eval()
    # if isinstance(val_loader.sampler, DistributedSampler):
    #     val_loader.sampler.set_epoch(0)

    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    is_main_rank = (rank == 0)

    pbar = tqdm(val_loader, disable=not is_main_rank, desc=f"[Eval@E{epoch}]", dynamic_ncols=True)

    n_seen = bbox_l1_sum = iou_mean_sum = v_bce_sum = ae_total_sum = 0.0
    cls_ce_sum = cls_acc_sum = n_seen_cls = 0.0

    scan_preds_path = os.path.join(args.save_ckp_path, f"eval_scanpath_step{global_step}.jsonl")
    caption_preds_path = os.path.join(args.save_ckp_path, f"eval_captions_step{global_step}.json")
    best_score_path = os.path.join(args.save_ckp_path, "best_caption_score.json")

    if args.is_fewshot:
        caption_preds_path = os.path.join(args.save_ckp_path, f"fewshot_stage_{args.stage}_seed_{args.seed}_sample_{args.num_samples}.json")
        best_score_path = os.path.join(args.save_ckp_path, "best_caption_score.json")

    scan_preds_f = open(scan_preds_path, "w", encoding="utf-8") if is_main_rank else None
    all_caption_results = []

    if is_main_rank:
        print_rank0("🚀 Generating subject embeddings from training set...")
    subj_emb_map = collect_subject_embeddings(model, sample_loader, device, args, subject_id_to_name)

    if is_main_rank:
        print_rank0("🚀 Generating image descriptions and scanpaths...")

    for i, batch in enumerate(pbar, start=1):
        if i > max_batches:
            break

        subj_batch = {k: (v.to(local_rank, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                      for k, v in batch["subj"].items()}
        qwen_batch = {k: (v.to(local_rank, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                      for k, v in batch["qwen"].items()}


        # inference
        qwen_outputs, ae_out = model.module.inference(batch, processor, device, subj_emb_map)
        # qwen_outputs, ae_out = model.module.inference_new(batch, processor, device, subj_emb_map, sample_loader)

        # ---- AE losses
        losses = ae_out["losses"]
        bbox_l1 = losses["bbox_l1"].detach().clone() if "bbox_l1" in losses else torch.tensor(0.0, device=device)
        iou_mean = losses["iou_mean"].detach().clone() if "iou_mean" in losses else torch.tensor(0.0, device=device)
        vb = losses["v_bce"].detach().clone() if "v_bce" in losses else torch.tensor(0.0, device=device)
        ae_tot = losses["total"].detach().clone() if "total" in losses else torch.tensor(0.0, device=device)
        if dist.is_initialized():
            for t in (bbox_l1, iou_mean, vb, ae_tot):
                dist.all_reduce(t, op=dist.ReduceOp.AVG)
        bbox_l1_sum += bbox_l1.item()
        iou_mean_sum += iou_mean.item()
        v_bce_sum += vb.item()
        ae_total_sum += ae_tot.item()
        n_seen += 1

        # ---- classification
        logits = ae_out.get("subj_logits", None)
        if logits is not None:
            probs = torch.softmax(logits, dim=-1)
            pred_ids = probs.argmax(dim=-1)
            subj_ids_tensor = subj_batch["subject_ids"].to(device, non_blocking=True)
            ce = F.cross_entropy(logits, subj_ids_tensor).detach().clone()
            acc = (pred_ids == subj_ids_tensor).float().mean().detach().clone()
            if dist.is_initialized():
                dist.all_reduce(ce, op=dist.ReduceOp.AVG)
                dist.all_reduce(acc, op=dist.ReduceOp.AVG)
            cls_ce_sum += ce.item()
            cls_acc_sum += acc.item()
            n_seen_cls += 1
        else:
            pred_ids = None

        # ---- collect caption results (batch-safe)
        if isinstance(qwen_outputs, list):
            all_caption_results.extend(qwen_outputs)
        else:
            all_caption_results.append(qwen_outputs)

        # ---- progress bar
        if is_main_rank:
            desc = (f"[Eval@E{epoch}] bbox_l1 {bbox_l1_sum/n_seen:.4f} | iou_mean {iou_mean_sum/n_seen:.4f} | "
                    f"v_bce {v_bce_sum/n_seen:.4f} | ae_total {ae_total_sum/n_seen:.4f}")
            if n_seen_cls > 0:
                desc += f" | cls_ce {cls_ce_sum/n_seen_cls:.4f} | cls_acc {cls_acc_sum/n_seen_cls:.4f}"
            pbar.set_description(desc)

        # ---- write scanpath predictions (main rank only)
        if is_main_rank:
            recon_scan = ae_out["recon"]["recon_scan"].detach().cpu().numpy() if ae_out["recon"] else None  # (B,T,3)
            img_names = subj_batch.get("image_names") or [None] * recon_scan.shape[0]
            subj_gt = subj_batch.get("subject_ids", torch.zeros(recon_scan.shape[0])).cpu().tolist()
            subj_pred = pred_ids.cpu().tolist() if pred_ids is not None else [None] * len(subj_gt)
            for bi in range(len(img_names)):
                record = {
                    "image_name": str(img_names[bi]),
                    "subject_gt": subj_gt[bi],
                    "subject_pred": subj_pred[bi],
                    "scanpath": recon_scan[bi].tolist() if recon_scan is not None else None,
                }
                scan_preds_f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # ---- aggregate metrics across ranks
    if dist.is_initialized():
        sums = torch.tensor(
            [bbox_l1_sum, iou_mean_sum, v_bce_sum, ae_total_sum, cls_ce_sum, cls_acc_sum, n_seen, n_seen_cls],
            device=device,
        )
        dist.all_reduce(sums, op=dist.ReduceOp.SUM)
        bbox_l1_sum, iou_mean_sum, v_bce_sum, ae_total_sum, cls_ce_sum, cls_acc_sum, n_seen, n_seen_cls = sums.tolist()
    # ---- rank 0 post-processing
    # ---- gather all_caption_results from all ranks
    if dist.is_initialized():
        gathered_lists = [None for _ in range(world_size)]
        dist.gather_object(all_caption_results, gathered_lists if is_main_rank else None)
    else:
        gathered_lists = [all_caption_results]

    # ---- only rank 0 merges and saves
    if is_main_rank:
        # flatten gathered lists
        merged_caption_results = []
        for sublist in gathered_lists:
            if sublist is not None:
                merged_caption_results.extend(sublist)

        # sort to put one-image-mutliple-captions results first at the beginning
        by_image = defaultdict(list)
        for rec in merged_caption_results:
            by_image[rec["id"]].append(rec)

        # separate multi- and single-subject groups
        multi_groups = {k: v for k, v in by_image.items() if len(v) > 1}
        single_groups = {k: v for k, v in by_image.items() if len(v) == 1}

        # flatten: multi first, then singles
        sorted_records = [r for records in multi_groups.values() for r in records] + \
                        [r for records in single_groups.values() for r in records]
        scan_preds_f.close()

        print_rank0(f"📄 Wrote eval predictions to {scan_preds_path}")
        with open(caption_preds_path, "w") as f:
            json.dump(sorted_records, f, indent=2)
        print_rank0(f"💾 Saved caption results from all {world_size} GPUs to {caption_preds_path}")

        # caption + subject metrics
        all_true, all_pred = [], []
        for res in all_caption_results:
            if "true_subject_id" in res and "predicted_subject_id" in res:
                all_true.append(res["true_subject_id"])
                all_pred.append(res["predicted_subject_id"])

        metrics_cls = compute_emb_score(all_true, all_pred)
        metrics_caption = compute_caption_score(caption_preds_path)

        # weighted score
        weighted_score = (
            0.1 * metrics_caption.get("BLEU_1", 0.0)
            + 0.2 * metrics_caption.get("BLEU_4", 0.0)
            + 0.1 * metrics_caption.get("ROUGE_L", 0.0)
            + 0.2 * metrics_caption.get("METEOR", 0.0)
            + 0.2 * metrics_caption.get("CIDEr", 0.0)
        )
        best_score = 0.0
        if os.path.exists(best_score_path):
            with open(best_score_path, "r") as f:
                best_state = json.load(f)
            best_score = best_state.get("best_score", 0.0)
        else:
            best_state = {}

        if weighted_score > best_score:
            best_state = {
                "epoch": epoch,
                "global_step": global_step,
                "best_score": weighted_score,
                "metrics_caption": metrics_caption,
                "metrics_cls": metrics_cls,
            }
            with open(best_score_path, "w") as f:
                json.dump(best_state, f, indent=2)
            print_rank0(f"🏆 New best score {weighted_score:.4f} (prev {best_score:.4f})")
            save_state(model, optimizer, epoch, global_step, args)

    # --- make sure all ranks receive sorted_records
    if dist.is_initialized():
        obj = [sorted_records] if is_main_rank else [None]
        dist.broadcast_object_list(obj, src=0)
        sorted_records = obj[0]

    bbox_l1_mean = bbox_l1_sum / max(n_seen, 1)
    iou_mean = iou_mean_sum / max(n_seen, 1)
    v_bce_mean = v_bce_sum / max(n_seen, 1)
    ae_total_mean = ae_total_sum / max(n_seen, 1)
    cls_ce_mean = cls_ce_sum / max(n_seen_cls, 1)
    cls_acc_mean = cls_acc_sum / max(n_seen_cls, 1)

    total_with_cls = ae_total_mean + args.lambda_subj * cls_ce_mean

    metrics_recon = {
        "bbox_l1": bbox_l1_mean,
        "iou_mean": iou_mean,
        "v_bce": v_bce_mean,
        "cls_acc": cls_acc_mean,
    }
    if is_main_rank:
        all_metrics = (
            metrics_caption | metrics_recon | metrics_cls
            | {"epoch": epoch, "global_step": global_step}
        )
        print(all_metrics)

    # --- make sure all ranks receive all_metrics
    if dist.is_initialized():
        obj = [all_metrics] if is_main_rank else [None]
        dist.broadcast_object_list(obj, src=0)
        all_metrics = obj[0]
    
    return all_metrics, sorted_records




# @torch.no_grad()
# def collect_subject_embeddings(model, dataloader, device, args, subject_id_to_name):
#     """
#     Collect and average subject embeddings across all ranks (DDP-safe).

#     Returns:
#         subj_emb_map: {subject_name: torch.Tensor(D)}
#     """
#     model.eval()
#     # if isinstance(dataloader.sampler, DistributedSampler):
#     #     dataloader.sampler.set_epoch(0)
#     num_subjects = args.num_subjects
#     emb_dim = args.sen_hidden_size
#     rank = dist.get_rank() if dist.is_initialized() else 0
#     world_size = dist.get_world_size() if dist.is_initialized() else 1
#     is_main_rank = (rank == 0)

#     emb_sum = torch.zeros(num_subjects, emb_dim, device=device)
#     emb_count = torch.zeros(num_subjects, device=device)

#     # ------------------------------
#     # accumulate embeddings per rank
#     # ------------------------------

#     for batch in tqdm(dataloader, desc=f"[Rank {rank}] Collecting embeddings", disable=not is_main_rank):
#         subj_batch = batch["subj"]

#         ae_out = model.module.subj_net(
#             images=subj_batch["images"],
#             captions=subj_batch["captions"],
#             scanpaths=subj_batch["scanpaths"],
#             reconstruct=False,
#         )

#         z_subj = ae_out["z_subj"]                     # (B, D)
#         subj_ids = subj_batch["subject_ids"].to(device)  # (B,)
#         emb_sum.index_add_(0, subj_ids, z_subj)
#         emb_count.index_add_(0, subj_ids, torch.ones_like(subj_ids, dtype=torch.float))

#     # ------------------------------
#     # reduce across GPUs
#     # ------------------------------
#     if dist.is_initialized():
#         dist.all_reduce(emb_sum, op=dist.ReduceOp.SUM)
#         dist.all_reduce(emb_count, op=dist.ReduceOp.SUM)

#     # avoid div-by-zero
#     emb_count = emb_count.clamp(min=1e-6).unsqueeze(1)
#     subj_avg = emb_sum / emb_count  # (num_subjects, D)

#     # ------------------------------
#     # gather subject_id_to_name mappings across ranks
#     # ------------------------------
#     local_name_map = subject_id_to_name
#     gathered_name_maps = [None for _ in range(world_size)] if is_main_rank else None
#     if dist.is_initialized():
#         dist.gather_object(local_name_map, gathered_name_maps)
#     else:
#         gathered_name_maps = [local_name_map]

#     # merge all name maps on rank 0
#     merged_name_map = {}
#     if is_main_rank:
#         for m in gathered_name_maps:
#             if m is not None:
#                 merged_name_map.update(m)
#     else:
#         merged_name_map = None

#     # broadcast merged map so every rank has the same view
#     if dist.is_initialized():
#         obj_list = [merged_name_map]
#         dist.broadcast_object_list(obj_list, src=0)
#         merged_name_map = obj_list[0]

#     # ------------------------------
#     # build final dictionary
#     # ------------------------------
#     subj_emb_map = {}
#     for subj_idx, subj_vec in enumerate(subj_avg):
#         name = merged_name_map.get(subj_idx, f"subject_{subj_idx}")
#         subj_emb_map[name] = subj_vec

#     # cross-subject embedding shuffling (ablation)
#     # all_names = list(subj_emb_map.keys())
#     # shuffled_names = all_names.copy()
#     # random.shuffle(shuffled_names)
#     # shuffled_map = {orig: subj_emb_map[shuf] for orig, shuf in zip(all_names, shuffled_names)}
#     # subj_emb_map = shuffled_map


#     # save individual embeddings per subject
#     subj_emb_map = {
#         subject_id_to_name.get(i, f"subject_{i}"): subj_avg[i]
#         for i in range(num_subjects)
#     }

#     return subj_emb_map



@torch.no_grad()
def collect_subject_embeddings(model, dataloader, device, args, subject_id_to_name, save_dir=None):
    """
    Collect all subject embeddings (z_subj) across all ranks and save them into one file.
    Works safely with DDP.
    Returns:
        subj_emb_map: {subject_name: averaged_embedding (D,)}
    """
    model.eval()
    num_subjects = args.num_subjects
    emb_dim = args.sen_hidden_size

    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    is_main_rank = (rank == 0)

    # Accumulate embeddings and counts
    emb_sum = torch.zeros(num_subjects, emb_dim, device=device)
    emb_count = torch.zeros(num_subjects, device=device)

    # Store individual embeddings per subject (local to this rank)
    all_subj_embs = {i: [] for i in range(num_subjects)}

    for batch in tqdm(dataloader, desc=f"[Rank {rank}] Collecting embeddings", disable=not is_main_rank):
        subj_batch = batch["subj"]
        ae_out = model.module.subj_net(
            images=subj_batch["images"],
            captions=subj_batch["captions"],
            scanpaths=subj_batch["scanpaths"],
            reconstruct=False,
        )

        z_subj = ae_out["z_subj"]                     # (B, D)
        subj_ids = subj_batch["subject_ids"].to(device)  # (B,)

        for z, sid in zip(z_subj, subj_ids):
            all_subj_embs[int(sid)].append(z.detach().cpu())

        emb_sum.index_add_(0, subj_ids, z_subj)
        emb_count.index_add_(0, subj_ids, torch.ones_like(subj_ids, dtype=torch.float))

    # --- Reduce averages across ranks ---
    if dist.is_initialized():
        dist.all_reduce(emb_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(emb_count, op=dist.ReduceOp.SUM)
    subj_avg = emb_sum / emb_count.clamp(min=1e-6).unsqueeze(1)

    # --- Gather all individual embeddings across ranks ---
    if dist.is_initialized():
        gathered_embs = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_embs, all_subj_embs)

        # if is_main_rank:
        #     merged_embs = {i: [] for i in range(num_subjects)}
        #     for rank_embs in gathered_embs:
        #         if rank_embs is None:
        #             continue
        #         for sid, lst in rank_embs.items():
        #             merged_embs[sid].extend(lst)
        #     all_subj_embs = merged_embs

    # --- Save only once on rank 0 ---
    # if is_main_rank:
    #     all_embeddings = {
    #         subject_id_to_name.get(i, f"subject_{i}"): torch.stack(embs)
    #         for i, embs in all_subj_embs.items() if len(embs) > 0
    #     }
    #     save_path = os.path.join(args.save_ckp_path, "all_subject_embeddings_unseen.pt")
    #     torch.save(all_embeddings, save_path)
    #     print(f"✅ Saved all subject embeddings from all ranks to {save_path}")

    # --- Build and return averaged map ---
    subj_emb_map = {
        subject_id_to_name.get(i, f"subject_{i}"): subj_avg[i]
        for i in range(num_subjects)
    }

    return subj_emb_map
