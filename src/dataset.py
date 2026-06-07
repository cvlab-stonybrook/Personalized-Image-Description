import os
import random
import torch

import torch.distributed as dist

from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from utils import print_rank0


class SubjectQwenDataset(Dataset):
    def __init__(self, args, data_list, image_path, processor, split,
                 max_caption_len=128, excluded_subjects=None, fewshot_subjects=None):
        """
        Args:
            data_list: list of dicts with keys ["name", "caption", "subject", "subject_name", "X", "Y", "T"]
            image_path: folder containing images
            tokenizer: tokenizer for captions
            excluded_subjects: list or set of subject ids to exclude
            fewshot_subjects: list or set of subject ids to test few-shot performance
        """
        self.args = args
        self.image_path = image_path
        self.processor = processor
        self.max_caption_len = max_caption_len
        excluded_subjects = set(excluded_subjects) if excluded_subjects else set()
        self.split = split

        # if the current mode is few-shot inference, use the subject id in excluded_subjects
        # as few-shot subject ids. 
        if args.is_fewshot:
            fewshot_subjects = excluded_subjects
            excluded_subjects = None


        # filter data
        if fewshot_subjects is not None:
            self.data_list = [s for s in data_list if s["subject"] in fewshot_subjects]
        elif excluded_subjects is not None:
            self.data_list = [s for s in data_list if s["subject"] not in excluded_subjects]

        # sample num_fewshot_sample images per few-shot subject as support set
        if fewshot_subjects is not None and split == 'train':
            fewshot_data = []
            for sid in fewshot_subjects:
                subj_samples = [s for s in self.data_list if s["subject"] == sid]
                if len(subj_samples) <= args.num_fewshot_sample:
                    fewshot_data.extend(subj_samples)
                else:
                    fewshot_data.extend(random.sample(subj_samples, args.num_fewshot_sample))
            self.data_list = fewshot_data
    


        # group samples by subject
        self.by_subject = {}
        for s in self.data_list:
            sid = s["subject"]
            self.by_subject.setdefault(sid, []).append(s)

        print_rank0(f"Dataset initialized for split '{split}' with {len(self.data_list)} samples")

        self.system_message = """You are a helpful assistant."""

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        sample = self.data_list[idx]

        # -------------------
        # Subject embedding input
        # -------------------
        img_name = sample["name"]
        caption = sample["caption"]
        subject_id = sample["subject"]
        subject_name = sample["subject_name"]

        scan_x = sample["X"]
        scan_y = sample["Y"]
        scan_t = [int(round(float(t) * 100)) for t in sample["T"]]
        bboxes = sample["bboxes"]  # normalized [0, 1], list of [x1, y1, x2, y2]


        img_path = os.path.join(self.image_path, img_name)
        image = Image.open(img_path).convert("RGB")
        image = image.resize((448, 448), Image.Resampling.BICUBIC)

        subj_inputs = {
            "image_name": img_name,
            "image": image,
            # "image_feat": image_feat,
            "caption": caption,
            "subject": subject_id,
            "subject_name": subject_name,
            "scan_x": torch.tensor(scan_x, dtype=torch.float32),
            "scan_y": torch.tensor(scan_y, dtype=torch.float32),
            "scan_t": torch.tensor(scan_t, dtype=torch.float32),
            "bboxes": torch.tensor(bboxes, dtype=torch.float32)
        }

        # -------------------
        # Qwen2 random pair (same subject)
        # -------------------
        if self.split == 'train':
            same_subject_samples = self.by_subject[subject_id]
            img_cap_sample = random.choice(same_subject_samples)

            img2_path = os.path.join(self.image_path, img_cap_sample["name"])
            image2 = img2_path
            image2 = Image.open(img2_path).convert("RGB")
            caption2 = img_cap_sample["caption"]
            subject_name2 = img_cap_sample["subject_name"]
        elif self.split == 'val':
            img2_path = img_path
            image2 = img2_path
            image2 = Image.open(img2_path).convert("RGB")
            caption2 = caption
            subject_name2 = sample["subject_name"]

        w, h = image2.size
        image2 = image2.resize((448, 448), Image.Resampling.BICUBIC)

        messages = [
            {"role": "system", "content": [{"type": "text", "text": self.system_message}]},
            {"role": "user", "content": [
                {"type": "image", "image": image2},
                {"type": "text", "text": f"<image>\nWrite a detailed description for this photo in the style of {subject_name2}."},
                # use the prompt below if you are running DKollenda or SenHe dataset
                # {"type": "text", "text": f"<image>\nCaption this image in the style of {subject_name2}."},
                {"type": "subject", "name": subject_name2},
                {"type": "image_path", "path": img2_path}
            ]},
            {"role": "assistant", "content": [{"type": "text", "text": caption2}]}
        ]
        qwen_inputs = {
            "messages": messages,
            "subject_id": subject_id,
            "subject_name": subject_name2,
        }

        return {"subj_inputs": subj_inputs, "qwen_inputs": qwen_inputs}


def collate_fn(batch, processor, max_scan_length=150, max_text_length=256):
    tokenizer = processor.tokenizer
    processor.tokenizer.padding_side = "right"

    # ======================
    # Subject embedding inputs
    # ======================
    subject_ids = torch.tensor([b["subj_inputs"]["subject"] for b in batch], dtype=torch.long)
    subject_names = [b["subj_inputs"]["subject_name"] for b in batch]

    images = [b["subj_inputs"]["image"] for b in batch]   # list of PILs
    image_names = [b["subj_inputs"]["image_name"] for b in batch]
    # image_feats = torch.cat([b["subj_inputs"]["image_feat"] for b in batch], dim=0)  # (B, 1, 768)

    # pad scanpaths
    Xs, Ys, Ts, bboxes, masks = [], [], [], [], []
    for b in batch:
        L = len(b["subj_inputs"]["scan_x"])

        if L >= max_scan_length:
            # truncate
            Xs.append(b["subj_inputs"]["scan_x"][:max_scan_length])
            Ys.append(b["subj_inputs"]["scan_y"][:max_scan_length])
            Ts.append(b["subj_inputs"]["scan_t"][:max_scan_length])
            bboxes.append(b["subj_inputs"]["bboxes"][:max_scan_length])
            masks.append(torch.zeros(max_scan_length, dtype=torch.bool))  # all valid
        else:
            # pad
            pad_len = max_scan_length - L
            Xs.append(torch.cat([b["subj_inputs"]["scan_x"], torch.zeros(pad_len)]))
            Ys.append(torch.cat([b["subj_inputs"]["scan_y"], torch.zeros(pad_len)]))
            Ts.append(torch.cat([b["subj_inputs"]["scan_t"], torch.zeros(pad_len)]))
            bboxes.append(torch.cat([b["subj_inputs"]["bboxes"], torch.zeros((pad_len, 4))]))
            masks.append(torch.cat([
                torch.zeros(L, dtype=torch.bool),
                torch.ones(pad_len, dtype=torch.bool)
            ]))

    scanpath_batch = {
        "X": torch.stack(Xs),
        "Y": torch.stack(Ys),
        "T": torch.stack(Ts),
        "bboxes": torch.stack(bboxes),
        "mask": torch.stack(masks),
    }

    # tokenize subject captions
    subj_captions = [b["subj_inputs"]["caption"] for b in batch]
    cap_enc = tokenizer(
        subj_captions,
        padding="longest",
        truncation=True,
        max_length=max_text_length,
        return_tensors="pt"
    )

    # ======================
    # Qwen inputs
    # ======================
    qwen_messages = [b["qwen_inputs"]["messages"] for b in batch]
    qwen_subjects_ids = torch.tensor([b["qwen_inputs"]["subject_id"] for b in batch], dtype=torch.long)
    qwen_subject_names = [b["qwen_inputs"]["subject_name"] for b in batch]

    return {
        "subj": {
            "image_names": image_names,
            "images": images,
            # "image_feats": image_feats,
            "captions": cap_enc,
            "subject_ids": subject_ids,
            "subject_names": subject_names,
            "scanpaths": scanpath_batch,
        },
        "qwen": {
            "messages": qwen_messages,
            "subject_ids": qwen_subjects_ids,
            "subject_names": qwen_subject_names,
        }
    }