export DS_SKIP_CUDA_CHECK=1
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
deepspeed --include localhost:4,5,6,7 --master_port 13903 /data1/dcy/projects/fine-tune/fine-tune-yyz/train/train.py \
    --model_path /data1/dcy/downloads/model/meta-llama/Meta-Llama-3-8B-Instruct \
    --save_name llama3_shuffle \
    --max_epoch 10 \
    --max_steps 200000 \
    --save_steps 1000 \
