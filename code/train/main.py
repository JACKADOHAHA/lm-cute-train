# # import json
# # import random
# # import pickle
# # from tqdm import tqdm

# # # 读取并解析JSONL文件
# # def read_jsonl(file_path):
# #     data = []
# #     with open(file_path, "r") as f:
# #         for line in f:
# #             data.append(json.loads(line))
# #     return data

# # # 主逻辑
# # def main():
# #     data_llama7b = read_jsonl("/data1/dcy/vllm/llama2/output/llama--Llama-2-7b-chat-hf_0-shot/inference/llama--Llama-2-7b-chat-hf.jsonl")
# #     data_wizard = read_jsonl("/data1/dcy/refine-llama2-7b-20240315/llama_20230217_13b_tryans2/data/20230205_llama2test/wizard.jsonl")
# #     fp = open("refine_wizard_raw.jsonl", "w")
# #     n = len(data_llama7b)
# #     for i in tqdm(range(n)):

# #         question = data_wizard[i]["question"]
# #         answer = data_wizard[i]["output"]
# #         openchat3b_ans = data_llama7b[i]["openchat_3b"]
# #         dct = {
# #             "question": question,
# #             "answer": answer,
# #             "openchat_3b": openchat3b_ans,
# #         }
# #         print(json.dumps(dct, ensure_ascii=False), file=fp)
# #     fp.close()
        

# # if __name__ == "__main__":
# #     main()
import json
import random
import pickle
from tqdm import tqdm

# 读取并解析JSONL文件
def read_jsonl(file_path):
    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data






# def merge_datasets(dataset1, dataset2):
#     # 创建一个空字典来存储基于question的合并数据
#     merged_data = {}
    
#     # 首先处理第一个数据集
#     for item in dataset1:
#         question = item['question']  # 假设每个条目都有一个'question'键
#         if question not in merged_data:
#             merged_data[question] = {'data1': item}
#         else:
#             merged_data[question].update({'data1': item})
    
#     # 然后处理第二个数据集
#     for item in dataset2:
#         question = item['question']
#         if question not in merged_data:
#             merged_data[question] = {'data2': item}
#         else:
#             merged_data[question].update({'data2': item})
    
#     # 将字典转换为列表格式
#     merged_list = []
#     for question, data in merged_data.items():
#         merged_list.append({
#             'question': question,
#             'data1': data.get('data1'),
#             'data2': data.get('data2')
#         })
    
#     return merged_list


# # 主逻辑
# def main():
#     data_llama7b = read_jsonl("/data1/dcy/vllm/llama2/output/llama--Llama-2-7b-chat-hf_0-shot/inference/llama--Llama-2-7b-chat-hf.jsonl")
#     moe1 = read_jsonl("/data1/dcy/finished/inference_all_vllm/inference_vllm_moe/vllm_moe_wizard_5000_10000.jsonl")
#     moe2 = read_jsonl("/data1/dcy/finetune_gpt/data/20240227_moe_ift/vllm_moe_wizard.jsonl")
#     moe3 = read_jsonl("/data1/dcy/inference_vllm_moe/vllm_moe_wizard20240227_moe_ift_10000_20000.jsonl")
#     moe4 = read_jsonl("/data1/dcy/inference_vllm_moe/vllm_moe_wizard20240227_moe_ift_40000_50000.jsonl")

#     # print(len(moe1))
#     # print(len(moe2))
#     # print(len(moe3))
#     # print(len(moe4))
#     moe5 = moe1 + moe2 + moe3 + moe4
#     # print(len(moe5))
#     # print(moe5[:1])
#     # print(len(data_llama7b))
#     # print(data_llama7b[:1])


#         # 假设data_llama7b和moe5是你已经加载的数据集
#     merged_datasets = merge_datasets(data_llama7b, moe5)
    
#     # 打印合并后的数据集的长度和前几个元素进行检查
#     print(len(merged_datasets))
#     print(merged_datasets[:1])
#     formatted_json = json.dumps(merged_datasets, indent=4, ensure_ascii=False)
#     print(formatted_json[:1])
# if __name__ == "__main__":
#     main()
import json

# 假设read_jsonl是一个用于读取.jsonl文件并返回数据列表的函数

def merge_datasets(data_llama7b, moe_datasets):
    moe_dict = {item['question']: item for item in moe_datasets}  # 创建以question为键的字典
    merged_data = []
    for item in data_llama7b:
        question = item['question']
        if question in moe_dict:  # 检查是否存在于moe数据集
            merged_item = {
                'question': question,
                'llama7b_ans': item['model_generate'],
                'moe_ans': moe_dict[question]['ans_moe'],
                'output':item['output_data'],
            }
            merged_data.append(merged_item)
    return merged_data

def save_to_file(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data
def main():
    # data_llama7b = read_jsonl("/data1/dcy/vllm/llama2/output/llama--Llama-2-7b-chat-hf_0-shot/inference/llama--Llama-2-7b-chat-hf.jsonl")
    # moe1 = read_jsonl("/data1/dcy/finished/inference_all_vllm/inference_vllm_moe/vllm_moe_wizard_5000_10000.jsonl")
    # moe2 = read_jsonl("/data1/dcy/finetune_gpt/data/20240227_moe_ift/vllm_moe_wizard.jsonl")
    # moe3 = read_jsonl("/data1/dcy/inference_vllm_moe/vllm_moe_wizard20240227_moe_ift_10000_20000.jsonl")
    # moe4 = read_jsonl("/data1/dcy/inference_vllm_moe/vllm_moe_wizard20240227_moe_ift_40000_50000.jsonl")

    # moe5 = moe1 + moe2 + moe3 + moe4
    # merged_datasets = merge_datasets(data_llama7b, moe5)
    
    # # 保存合并后的数据集到文件
    # save_to_file(merged_datasets, "merged_data.json")
    merged_data = read_json("/data1/dcy/refine-llama2-7b-20240315/yyz/train/merged_data.json")

    print(len(merged_data))
    print(merged_data[:1])
if __name__ == "__main__":
    main()
