# train.py
# -*- coding: utf-8 -*-

import os
import json
import time
import copy
import argparse
import random

import torch
import numpy as np
import torch.distributed as dist
import torch.nn.functional as F

from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
# from peft import set_peft_model_state_dict
from transformers import Qwen2VLProcessor, Qwen2VLForConditionalGeneration


from dataset import SubjectQwenDataset, collate_fn
from model import ScanAutoEncoder, ScanQwenModel_AE
from utils import (get_special_token_list_SenHe, setup_seed, setup_ddp, is_main, save_state, 
                   train_senet_one_epoch, eval_senet_greedy,
                   get_special_token_list_coco, get_special_token_list_flickr30k, get_special_token_list_population,
                   get_special_token_list_DKollenda, 
                   init_peft, get_trainable_state_dict, print_rank0,
                   joint_train_senet_qwen_one_epoch, eval_qwen_senet_greedy)
from loss import supervised_contrastive_loss, ddp_all_gather_no_grad, SubjectMemoryBank

os.environ["TOKENIZERS_PARALLELISM"] = "false"

torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(False)

# ---------------------------
# Main
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="Train parallel scanpath AE (z_subj, z_scan)")
    # Data roots
    parser.add_argument("--server_ip", type=str, default="/nfs/130.245.4.102")
    parser.add_argument("--train_image_path", type=str, default="data/labels/coco/train2017")
    parser.add_argument("--val_image_path",   type=str, default="data/labels/coco/val2017")
    parser.add_argument("--train_json", type=str, default="data/labels/coco/scanpath/coco_scanpath_refine.json")
    parser.add_argument("--val_json",   type=str, default="data/labels/coco/scanpath/coco_scanpath_refine_sample_100.json")
    # Save / logs
    parser.add_argument("--save_ckp_path", type=str, required=True)
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--print_every", type=int, default=100)
    parser.add_argument("--eval_every", type=int, default=2000)
    parser.add_argument("--eval_batch_num", type=int, default=5000000)

    # Model hparams (must match model.py)
    parser.add_argument("--img_encoder_name", type=str, default="facebook/dinov3-convnext-tiny-pretrain-lvd1689m")
    parser.add_argument("--feat_h", type=int, default=34)
    parser.add_argument("--feat_w", type=int, default=46)
    # parser.add_argument("--feat_h", type=int, default=15)
    # parser.add_argument("--feat_w", type=int, default=15)
    parser.add_argument("--sen_hidden_size", type=int, default=384)
    parser.add_argument("--qwen_hidden_size", type=int, default=1536)
    parser.add_argument("--img_feat_dim", type=int, default=768)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num_subjects", type=int, default=29)

    parser.add_argument("--scan_enc_layers", type=int, default=2)
    parser.add_argument("--scan_dec_layers", type=int, default=4)
    parser.add_argument("--scan_max_len", type=int, default=30)
    parser.add_argument("--scan_valid_thresh", type=float, default=0.5)
    parser.add_argument("--subj_tokens", type=int, default=1)

    # Train hyperparams
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--grad_accum_steps", type=int, default=16)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--lambda_subj", type=float, default=1.0)  # weight for subj classification loss
    parser.add_argument("--lambda_contrast", type=float, default=0.1)  # weight for contrastive loss
    parser.add_argument("--contrast_temp", type=float, default=0.07)

    parser.add_argument("--senet_ckp", type=str, default="")
    parser.add_argument("--qwen_ckp", type=str, default="")  # for resuming full model training
    parser.add_argument("--stage", type=int, choices=[1, 2, 3, 4], required=True,
                    help="1=train ScanAutoEncoder only; 2=train qwen model using pretrained AE from stage 1; " \
                    "3=resume training; 4=qwen pt")

    parser.add_argument("--is_eval", type=int, default=0)
    parser.add_argument("--is_fewshot", type=int, default=0, help="0 for seen user evaluation; 1 for few-shot evaluation on unseen users")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of support samples per subject during evaluation")
    parser.add_argument(
        "--excluded_subjects", type=int, nargs='+', default=None,help="List of subject ids, pass like: --excluded_subjects 1 2 3"
    )
    parser.add_argument("--num_fewshot_sample", type=int, default=5, help="Number of few-shot samples per subject during evaluation")
    parser.add_argument(
        "--fewshot_subjects", type=int, nargs='+', default=None,
        help="List of subject ids, pass like: --fewshot_subjects 1 2 3")
    args = parser.parse_args()

    os.makedirs(args.save_ckp_path, exist_ok=True)

    # Logs
    if is_main() and not args.is_eval:
        with open(os.path.join(args.save_ckp_path, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=2)
    log_file_loss = os.path.join(args.save_ckp_path, "train_log.jsonl")

    # DDP / seed
    local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    setup_seed(args.seed)

    # Processor (tokenizer for collate_fn)
    processor = Qwen2VLProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

    # Load data JSONs
    with open(args.train_json, "r") as f:
        train_data = json.load(f)
    train_meta = [x for x in train_data if x.get("split", "train") == "train"]

    val_meta = [x for x in train_data if x.get("split", "val") == "val"]
    if not len(val_meta):  
        val_meta = [x for x in train_data if x.get("split", "test") in ("test", "val")]

    # sample_data used to generate subject embedding from training set
    with open(args.val_json, "r") as f:
        sample_data = json.load(f)
    sample_meta = [x for x in sample_data if x.get("split", "train") == "train"]

    # tag
    if 'userstudy' in args.save_ckp_path:
        val_meta = [x for x in sample_data if x.get("split", "val") == "val"]

    subject_id_to_name = {}
    for data in sample_meta:
        subject_id_to_name[int(data["subject"])] = data["subject_name"]

    # Datasets / Loaders
    train_ds = SubjectQwenDataset(
        args, train_meta, args.train_image_path, processor, split='train', excluded_subjects=args.excluded_subjects
    )
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=10,
        pin_memory=True,
        collate_fn=lambda x: collate_fn(x, processor, max_scan_length=args.scan_max_len),
        drop_last=False,
    )

    val_ds = SubjectQwenDataset(
        args, val_meta, args.val_image_path, processor, split='val', excluded_subjects=args.excluded_subjects
    )
    val_sampler = DistributedSampler(val_ds) if dist.is_initialized() else None
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=10,
        pin_memory=True,
        collate_fn=lambda x: collate_fn(x, processor, max_scan_length=args.scan_max_len),
        drop_last=False,
    )

    sample_ds = SubjectQwenDataset(
        args, sample_meta, args.train_image_path, processor, split='train', excluded_subjects=args.excluded_subjects
    )
    sample_sampler = DistributedSampler(sample_ds) if dist.is_initialized() else None
    sample_loader = DataLoader(
        sample_ds,
        batch_size=args.batch_size,
        sampler=sample_sampler,
        shuffle=False,
        num_workers=10,
        collate_fn=lambda x: collate_fn(x, processor, max_scan_length=args.scan_max_len)
    )
    # sample_loader = train_loader    # tag

    # MODEL
    print_rank0("🔧 Loading model and processor...")
    base_model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct", device_map=None, torch_dtype=torch.bfloat16)
    
    # add subject tokens into vocabulary
    elif 'coco' in args.train_image_path:
        special_tokens = get_special_token_list_coco()
    elif 'flickr30k' in args.train_image_path:
        special_tokens = get_special_token_list_flickr30k()
    elif 'SenHe' in args.train_image_path:
        special_tokens = get_special_token_list_SenHe()
    elif 'DKollenda' in args.train_image_path:
        special_tokens = get_special_token_list_DKollenda()
    
    # if 'userstudy' in args.save_ckp_path:
    #     special_tokens = get_special_token_list_userstudy()

    if args.is_fewshot:
        fewshot_special_tokens = []
        for i, token in enumerate(special_tokens):
            if int(token.split('_')[-1].split('>')[0]) in args.excluded_subjects:
                fewshot_special_tokens.append(token)
        special_tokens = fewshot_special_tokens

    old_vocab_size = copy.deepcopy(len(processor.tokenizer))
    num_added = processor.tokenizer.add_tokens(special_tokens, special_tokens=False)
    target_vocab_size = base_model.config.vocab_size
    base_model.resize_token_embeddings(target_vocab_size)
    # initialize model
    model = ScanQwenModel_AE(base_model, processor, device, args)
    model = model.to(device)
    # model = model.subj_net.to(device)   # for stage 1 only

    global_step = 0
    resume_epoch = 1


    print_rank0(f"🔧 Loading Qwen2-VL weights from {args.qwen_ckp}... and load senet weights from {args.senet_ckp}")
    qwen_ckp = torch.load(args.qwen_ckp, map_location="cpu")
    senet_ckp_subj_z = qwen_ckp['subj_net']
    senet_ckp_encoder = torch.load(args.senet_ckp, map_location="cpu")['subj_net']
    new_state_dict = {}
    for name, param in senet_ckp_encoder.items():
        # remove subj_net. prefix
        if name.startswith("subj_net."):
            new_name = name[len("subj_net."):]
            new_state_dict[new_name] = param
    model.subj_net.load_state_dict(new_state_dict, strict=False)
    for name, param in senet_ckp_subj_z.items():
        # remove subj_net. prefix
        if name.startswith("subj_net."):
            new_name = name[len("subj_net."):]
            new_state_dict[new_name] = param
    model.subj_net.load_state_dict(new_state_dict, strict=False)
    model.subj_to_qwen_proj.load_state_dict(qwen_ckp['subj_to_qwen_proj'])
    # set_peft_model_state_dict(model.qwen, qwen_ckp['lora_net'])
    global_step = qwen_ckp['global_step']
    resume_epoch = qwen_ckp['epoch']
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr
    )

    # Wrap DDP
    if dist.is_initialized():
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    print_rank0("Trainable parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print_rank0(name)
            # add to log file
            if is_main():
                with open(log_file_loss, "a") as f:
                    f.write(f"Trainable parameter: {name}\n")


    # =====================
    # initialize contrastive learning utils
    # =====================
    bank = SubjectMemoryBank(emb_dim=args.sen_hidden_size, max_negatives=4096, device=device)


    # run inference

    for name, param in model.module.named_parameters():
                    param.requires_grad = False
                metrics = eval_qwen_senet_greedy(
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
                    optimizer=optimizer,
                )
    

    # Clean up
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

