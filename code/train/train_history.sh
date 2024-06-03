export DS_SKIP_CUDA_CHECK=1
export CUDA_LAUNCH_BLOCKING=1
deepspeed --include localhost:6,7 --master_port 13023 /data1/dcy/projects/fine-tune/fine-tune-yyz/train/train.py \
    --model_path /data1/yyz/downloads/models/Llama-2-7b-chat-hf \
    --save_name Llama-2-7b-chat-hf-sva \
    --max_epoch 2000 \
    --max_steps 10000 \
    --save_steps 1000

export DS_SKIP_CUDA_CHECK=1
export CUDA_LAUNCH_BLOCKING=1
deepspeed --include localhost:4,5 --master_port 13032 /data1/dcy/projects/fine-tune/fine-tune-yyz/train/train.py \
    --model_path /data1/yyz/downloads/models/Llama-2-7b-chat-hf \
    --save_name Llama-2-7b-chat-hf-gsm \
    --max_epoch 2000 \
    --max_steps 10000 \
    --save_steps 2000