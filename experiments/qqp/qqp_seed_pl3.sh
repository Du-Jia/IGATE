MAX_LENGTH=128
# MAX_STEPS=200  # epches = 200/4 = 50
# epoches = 1000 / (32/BS) = 250 

K=16
seed=42
bs=8

TASK=QQP \
DATA_ROOT=FewGLUE_32dev \
MODEL_TYPE=roberta \
MODEL="roberta-large" \
EMBED_SIZE=1024 \
TASK_NAME=qqp \
BS=$bs \
LR=$lr \
INFO_TYPE="conv" \
PATTERN_ID=3 \
WARMUP=100 \
GAMMA_INIT=0.005 \
GAMMA_TRAINABLE="False" \
TAG="qqp_seed_0620_" \
REPET=3
pl=3
max_steps=200
scale=10

# for pl in 3 6 9 12 15 18 21
# for pl in 3 6 9 12
for seed in 10 12 21 42 87  # 20 * 4, 30 * 4, 40*4, 50*4, 100*4
    do
        for lr in 5e-5
            do
                python cli.py \
                --data_dir $DATA_ROOT/$TASK/$K-$seed \
                --model_type $MODEL_TYPE \
                --model_name_or_path /root/$MODEL \
                --embed_size $EMBED_SIZE \
                --task_name $TASK_NAME \
                --output_dir output/$TASK_NAME/$INFO_TYPE/$seed \
                --do_train \
                --do_eval \
                --pet_per_gpu_eval_batch_size 512 \
                --pet_per_gpu_train_batch_size $BS \
                --pet_gradient_accumulation_steps 1 \
                --pet_max_seq_length $MAX_LENGTH \
                --pet_max_steps $max_steps \
                --warmup_steps $(expr $max_steps / $scale) \
                --pattern_ids $pl \
                --pet_repetitions $REPET \
                --learning_rate $lr \
                --overwrite_output_dir \
                --gamma $GAMMA_INIT \
                --trainable_gamma $GAMMA_TRAINABLE \
                --info_type $INFO_TYPE \
                --eval_every_step 20 \
                --tag $TAG$seed \
                --use_conv
            done
    done