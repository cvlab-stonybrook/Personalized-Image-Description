train_image_path="data/labels/flickr30k/images"
val_image_path="data/flickr30k/images"
train_json="data/scanpath/flickr30k_scanpath_refine.json"
val_json="data/scanpath/flickr30k_scanpath_refine_sample_100.json"
# excluded_subjects = 32 0 6 1 2 3 4 5 7 8       37 subjects


# train_image_path="data/labels/google/train2017"
# val_image_path="data/labels/google/val2017"
# train_json="data/scanpath/coco_scanpath_refine.json"
# val_json="data/scanpath/coco_scanpath_refine.json"
# exclude_subject = 30 82 37 67 39 57 89 0 92 33 19 48 75 15 91 20 55 79 32     108 subjects


# train_image_path="data/labels/SenHe/images"
# val_image_path="data/labels/SenHe/images"
# train_json="data/scanpath/SenHe_scanpath_refine.json"
# val_json="data/scanpath/SenHe_scanpath_refine_sample_100.json"

# train_image_path="data/labels/DKollenda/images"
# val_image_path="data/labels/DKollenda/images"
# train_json="data/scanpath/DKollenda_scanpath_refine.json"
# val_json="data/scanpath/DKollenda_scanpath_refine_sample_100.json"


# export NCCL_P2P_DISABLE=1       # disable direct peer-to-peer
# export NCCL_IB_DISABLE=1        # no Infiniband
# export NCCL_SOCKET_IFNAME=lo    # use loopback for intra-node NCCL comms
# export NCCL_ASYNC_ERROR_HANDLING=1
# export NCCL_DEBUG=ERROR



ckp_path="path/you/want/to/save/your/results"
senet_ckp_path="path/of/stage-1/ckp"
qwen_ckp_path="path/of/stage-2/ckp"

CUDA_VISIBLE_DEVICES=0,1 \
    torchrun --standalone --nnodes=1 --nproc_per_node=2 \
    test_fewshot.py \
    --batch_size 64 \
    --save_ckp_path $ckp_path \
    --train_image_path $train_image_path \
    --val_image_path $val_image_path \
    --train_json $train_json \
    --val_json $val_json \
    --senet_ckp $senet_ckp_path \
    --qwen_ckp $qwen_ckp_path \
    --eval_every 50 \
    --save_every 50 \
    --print_every 4 \
    --num_epochs 20 \
    --grad_accum_steps 4 \
    --stage 3 \
    --server_ip "/data" \
    --is_fewshot 1 \
    --num_subjects 37 \
    --eval_batch_num 1000000 \
    --seed 46 \
    --num_fewshot_sample 5 \
    --excluded_subjects 32 0 6 1 2 3 4 5 7 8
