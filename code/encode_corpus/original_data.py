
import datetime
import json
import os

gemma_2b = []
with open("/data1/dcy/projects/inference/generate/output/5-2_17:49_gemma-2b-it_len_10000.json","r") as f:
    llama3_8b = json.load(f)

llama3_8b = []
with open("/data1/dcy/projects/inference/generate/output/4-30_02:37_Meta-Llama-3-8B-Instruct_len_10000.json","r") as f:
    llama3_8b = json.load(f)
llama2_13b = []
with open("/data1/dcy/projects/inference/generate/output/5-14_22:30_Llama-2-13b-chat-hf_len_10000.json", "r") as f:
    llama2_13b = json.load(f)
llama3_70b = []
with open("/data1/dcy/projects/inference/generate/output/4-30_16:02_Meta-Llama-3-70B-Instruct_len_10000.json","r") as f:
    llama3_70b = json.load(f)


vicuna_7b = []
with open("/data1/dcy/projects/inference/generate/output/4-29_15:13_vicuna-7b-v1.5_len_10000.json","r") as f:
    vicuna_7b = json.load(f)
vicuna_13b = []
with open("/data1/dcy/projects/inference/generate/output/4-29_15:57_vicuna-13b-v1.5_len_10000.json","r") as f:
    vicuna_13b = json.load(f)
vicuna_33b = []
with open("/data1/dcy/projects/inference/generate/output/4-29_17:30_vicuna-33b-v1.3_len_10000.json","r") as f:
    vicuna_33b = json.load(f)


wizard_output = []
with open("/data1/dcy/projects/inference/generate/output/wizard_10000.json","r") as f:
    wizard_output = json.load(f)
    
    
    
Qwen_1_B = []
with open("/data1/dcy/projects/inference/generate/output/4-30_01:07_Qwen-1_8B-Chat_len_10000.json","r") as f:
    Qwen_7B = json.load(f)
Qwen_7B = []
with open("/data1/dcy/projects/inference/generate/output/4-30_01:34_Qwen-7B-Chat_len_10000.json","r") as f:
    Qwen_7B = json.load(f)
Qwen_14B = []
with open("/data1/dcy/projects/inference/generate/output/4-30_20:59_Qwen-14B-Chat_len_10000.json","r") as f:
    Qwen_14B = json.load(f)

merge_data = []
model_name = llama2_13b
better_answer = llama3_70b
model_name_file = "llama2_13b_70b"
better_answer_file = "llama3_70b"
for i in range(len(model_name)):
    if len(model_name[i]["output"])<5 or len(better_answer[i]["output"])<5:
        continue 
    if model_name[i]["question"] == better_answer[i]["question"]:
        merge_data.append({
            "question": model_name[i]["question"], 
            "output1": model_name[i]["output"],
            "output2": better_answer[i]["output"]
        })
file_path = f"/data1/dcy/projects/fine-tune/fine-tune-yyz/encode_corpus/data/{model_name_file}/"

os.makedirs(file_path, exist_ok=True)
with open(f"{file_path}{model_name_file}_merge.json", "w") as f:
    json.dump(merge_data, f, indent=2, ensure_ascii=False)


t = datetime.datetime.now()
data_to_save = [{
    "datasets": "llama3_70b", 
    "length": 10000, 
    "output1": model_name_file, 
    "output2": better_answer_file, 
    "date": f"{t.month}-{t.day}_{t.hour:02d}:{t.minute:02d}",
    "note": "",
    "output_len":len(merge_data),
    "sample":merge_data[:1],
}]
# 使用with语句打开文件，确保文件会被正确关闭
with open(f"{file_path}{model_name_file}_info.json", "w", encoding="utf-8") as file:
    # 将数据转换为JSON格式，然后写入文件
    json.dump(data_to_save, file, ensure_ascii=False, indent=2)

print("信息已保存到文件:", file_path)