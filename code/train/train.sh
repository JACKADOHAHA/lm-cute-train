export DS_SKIP_CUDA_CHECK=1
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
deepspeed --include localhost:4,5,6,7 --master_port 13903 /data1/dcy/projects/fine-tune/fine-tune-yyz/train/train.py \
    --model_path /data1/dcy/downloads/model/meta-llama/Meta-Llama-3-8B-Instruct \
    --save_name llama3_shuffle \
    --max_epoch 10 \
    --max_steps 200000 \
    --save_steps 1000 \



# deepspeed --include localhost:6,7,4,5 --master_port 13995 /data1/dcy/projects/fine-tune/fine-tune-yyz/train/train.py \
#     --model_path /data1/yyz/downloads/models/NousResearch/Llama-2-7b-chat-hf \
#     --save_name llama2-gsm \
#     --max_epoch 1 \
#     --max_steps 2000 \
#     --save_steps 500

# deepspeed --include localhost:0,1 --master_port 13992 /data1/dcy/projects/fine-tune/fine-tune-yyz/train/train.py \
#     --model_path /data1/dcy/downloads/model/lmsys/vicuna-7b-v1.5 \
#     --save_name vicuna-7b \
#     --max_epoch 2000 \
#     --max_steps 10000 \
#     --save_steps 2000 \

# deepspeed --include localhost:1,6 --master_port 13993 /data1/dcy/projects/fine-tune/fine-tune-yyz/train/train.py \
#     --model_path /data1/dcy/downloads/model/lmsys/vicuna-13b-v1.5 \
#     --save_name vicuna-13b \
#     --max_epoch 2000 \
#     --max_steps 10000 \
#     --save_steps 2000


# deepspeed --include localhost:3,4,5,6 --master_port 13994 /data1/dcy/projects/fine-tune/fine-tune-yyz/train/train.py \
#     --model_path /data1/dcy/downloads/model/lmsys/vicuna-33b-v1.3 \
#     --save_name vicuna-33b \
#     --max_epoch 2000 \
#     --max_steps 10000 \
#     --save_steps 2000


# deepspeed --include localhost:0,2 --master_port 13994 /data1/dcy/projects/fine-tune/fine-tune-yyz/train/train.py \
#     --model_path /data1/dcy/downloads/model/Qwen/Qwen-7B-Chat \
#     --save_name qwen_7b \
#     --max_epoch 2000 \
#     --max_steps 5000 \
#     --save_steps 2500

# deepspeed --include localhost:3 --master_port 14991 /data1/dcy/projects/fine-tune/fine-tune-yyz/train/train.py \
#     --model_path /data1/dcy/downloads/model/Qwen/Qwen-1_8B-Chat \
#     --save_name qwen_14b \
#     --max_epoch 1 \
#     --max_steps 10000 \
#     --save_steps 1250
# deepspeed --include localhost:6,7 --master_port 13995 /data1/dcy/projects/fine-tune/fine-tune-yyz/train/train.py \
#     --model_path /data1/dcy/downloads/model/Qwen/Qwen1.5-14B-Chat \
#     --save_name qwen1_5_14b \
#     --max_epoch 1 \
#     --max_steps 250000 \
#     --save_steps 1250
