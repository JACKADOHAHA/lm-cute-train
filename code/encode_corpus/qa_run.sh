python encode_qa.py \
    --tokenizer_path "/data1/dcy/downloads/model/meta-llama/Meta-Llama-3-8B-Instruct" \
    --corpus llama3_alll \
    --process_cnt 32 \
    --generator_mode \
    --max_length 2048  \
    --tokens_per_file 100000000
