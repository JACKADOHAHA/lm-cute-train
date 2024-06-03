python encode_qa.py \
    --tokenizer_path "/data1/dcy/downloads/model/meta-llama/Meta-Llama-3-8B-Instruct" \
    --corpus llama3_alll \
    --process_cnt 32 \
    --generator_mode \
    --max_length 2048  \
    --tokens_per_file 100000000


# python encode_qa.py \
#     --tokenizer_path "/data1/dcy/downloads/model/lmsys/vicuna-7b-v1.5" \
#     --corpus vicuna-7b \
#     --process_cnt 8 \
#     --generator_mode \
#     --max_length 2048 \
#     --conversations_per_file 100000

# python encode_qa.py \
#     --tokenizer_path "/data1/dcy/downloads/model/lmsys/vicuna-13b-v1.5" \
#     --corpus vicuna-13b \
#     --process_cnt 8 \
#     --generator_mode \
#     --max_length 2048 \
#     --conversations_per_file 100000
# python encode_qa.py \
#     --tokenizer_path "/data1/dcy/downloads/model/lmsys/vicuna-33b-v1.3" \
#     --corpus vicuna-33b \
#     --process_cnt 8 \
#     --generator_mode \
#     --max_length 2048 \
#     --conversations_per_file 100000


# python encode_qa.py \
#     --tokenizer_path "/data1/dcy/downloads/model/meta-llama/Meta-Llama-3-8B-Instruct" \
#     --corpus llama3 \
#     --process_cnt 8 \
#     --generator_mode \
#     --max_length 2048 \
#     --conversations_per_file 100000

# python encode_qa.py \
#     --tokenizer_path "/data1/yyz/downloads/models/Llama-2-7b-chat-hf" \
#     --corpus llama2-gsm \
#     --process_cnt 8 \
#     --generator_mode \
#     --max_length 2048 \
#     --conversations_per_file 100000
# /data1/yyz/downloads/models/vicuna-7b-v1.5