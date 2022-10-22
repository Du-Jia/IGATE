MAX_LENGTH=128
MAX_STEPS=1000
# epoches = 1000 / (32/BS) = 250 
K=16
bs=8
pl=6
lr=5e-5

for seed in 10 12
    do
        TASK=MRPC \
        DATA_ROOT=FewGLUE_32dev \
        MODEL_TYPE=roberta \
        MODEL="roberta-large" \
        EMBED_SIZE=1024 \
        TASK_NAME=mrpc \
        BS=$bs \
        LR=$lr \
        SEED=$seed \
        INFO_TYPE="conv" \
        PATTERN_ID=$pl \
        WARMUP=100 \
        GAMMA_INIT=0.005 \
        GAMMA_TRAINABLE="False" \
        TAG="mrpc_conv_10_0617" \

        python cli.py \
        --data_dir $DATA_ROOT/$TASK/$K-$SEED \
        --model_type $MODEL_TYPE \
        --model_name_or_path /root/$MODEL \
        --embed_size $EMBED_SIZE \
        --task_name $TASK_NAME \
        --output_dir output/$TASK_NAME/conv/$seed \
        --seed $seed \
        --do_train \
        --do_eval \
        --pet_per_gpu_eval_batch_size 128 \
        --pet_per_gpu_train_batch_size $BS \
        --pet_gradient_accumulation_steps 1 \
        --pet_max_seq_length $MAX_LENGTH \
        --pet_max_steps $MAX_STEPS \
        --warmup_steps $WARMUP \
        --pattern_ids $pl \
        --pet_repetitions 3 \
        --learning_rate $LR \
        --overwrite_output_dir \
        --gamma $GAMMA_INIT \
        --trainable_gamma $GAMMA_TRAINABLE \
        --info_type $INFO_TYPE \
        --eval_every_step 100 \
        --tag $TAG \
        --use_conv
    done