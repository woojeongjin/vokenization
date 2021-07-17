GPUS=$1
# The name of experiment
NAME=$2

# Create dirs and make backup
output=snap/vlpretrain/$NAME
mkdir -p $output/src/
cp -r vlpretrain $output/src/
cp $0 $output/run.bash

# Pre-training
# CUDA_VISIBLE_DEVICES=$GPUS unbuffer python xmatching/main.py \
#     --train-imgs mscoco_train,mscoco_nominival --valid-imgs mscoco_minival \
#     --train-langs mscoco --valid-langs mscoco \
#     --max-len 20 --dim 64 \
#     --lang-layers 4,3,2,1 \
#     --lang-pretrained --visn-pretrained \
#     --num-workers 8 --batchSize 256 --optim adam --lr 1e-3 --epochs 20 \
#     --nodes 1 --nr 0 \
#     --output $output ${@:3} | tee $output/log.log

#--visn resnext101_32x8d --lang bert \


# for our new idea
CUDA_VISIBLE_DEVICES=$GPUS python vlpretrain/main.py \
    --train-imgs mscoco_train,mscoco_nominival --valid-imgs mscoco_minival \
    --train-langs mscoco --valid-langs mscoco \
    --max-len 32 --dim 768 \
    --lang-finetune \
    --lang-pretrained --visn-pretrained \
    --num-workers 8 --batchSize 64 --optim adam  --epochs 10 \
    --nodes 1 --nr 0 \
    --output $output ${@:3} | tee $output/log.log

# for slurm
# python vlpretrain/main.py \
#     --train-imgs mscoco_train,mscoco_nominival --valid-imgs mscoco_minival \
#     --train-langs mscoco --valid-langs mscoco \
#     --max-len 32 --dim 768 \
#     --lang-finetune \
#     --lang-pretrained --visn-pretrained \
#     --num-workers 8 --batchSize 64 --optim adam  --epochs 20 \
#     --nodes 1 --nr 0 \
#     --output $output ${@:3} | tee $output/log.log



# bash scripts/run_vlpretrain.bash 0,1 bert_vector_1e-5 --visn vector --lang bert --lr 1e-5