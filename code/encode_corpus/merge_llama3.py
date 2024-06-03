
import datetime
import json
import os
import glob

ans1_paths = glob.glob('/data1/dcy/projects/evaluate/lm-cute-eval/output/5-25_00:25_llama3_gen/infer_results/*/*.json')
ans2_paths = glob.glob('/data1/dcy/projects/evaluate/lm-cute-eval/output/5-25_02:38_llama3_gen/infer_results/*/*.json')




merge_data = []
choice_data = ["mmlu", "arc", "hellaswag", "commonsenseqa", "winogrande"]
question_data = [ "gsm8k",  "drop", "xsum"]
code_data = ["humaneval"]
for ans1_path, ans2_path in zip(ans1_paths, ans2_paths):
    with open(ans1_path) as f:
        ans1 = json.load(f)
    with open(ans2_path) as f:
        ans2 = json.load(f)
    parts1 = ans1_path.split('/')
    data_name1 = parts1[-2]
    parts2 = ans2_path.split('/')
    data_name2 = parts2[-2]
    if data_name1 == "xsum":
        continue
    if data_name1 != data_name2:
        print(f"check dataset error at {data_name1} and {data_name2}")
    for model_name ,better_answer in zip(ans1, ans2):
        if len(model_name["infer_round1"])<10 or len(better_answer["infer_round1"])<10:
            continue
        print(data_name1)
        if not model_name["judge1"]  and better_answer["judge1"]:
            merge_data.append({
                "question": model_name["prompt_round1"], 
                "output1": model_name["infer_round1"],
                "output2": better_answer["infer_round1"]
            })
print(len(merge_data))
                

file_path = f"/data1/dcy/projects/fine-tune/fine-tune-yyz/encode_corpus/data/my_merge_data_llama2"

os.makedirs(file_path, exist_ok=True)
with open(f"{file_path}_merge.json", "w") as f:
    json.dump(merge_data, f, indent=2, ensure_ascii=False)

