import glob
import os
import pickle
import random

import numpy as np



# 假设 file_name 是您的.pkl文件的路径
file_names = ["/data1/dcy/projects/fine-tune/fine-tune-yyz/encode_corpus/output/llama3_refine/llama3_refine_0.pkl", "/data1/dcy/projects/fine-tune/fine-tune-yyz/encode_corpus/output/llama3_base/llama3_base_0.pkl"]

input_ids_list = []
labels_list = []

for file_name in file_names:
    # 打开文件并读取数据
    print(file_name)
    with open(file_name, "rb") as f:
        while True:
            try:
                input_ids, labels = pickle.load(f)
                input_ids_list.append(input_ids)
                labels_list.append(labels)
            except EOFError:
                break
dtype = np.int32  # 假设数据类型为np.int16，根据您的实际情况进行调整
input_ids_array = np.array(input_ids_list, dtype=dtype)
labels_array = np.array(labels_list, dtype=dtype)
# 创建一个列表，包含(input_ids, labels)对
data_pairs = list(zip(input_ids_array, labels_array))

# 洗牌列表
random.shuffle(data_pairs)

# 设置输出目录
output_dir = "/data1/dcy/projects/fine-tune/fine-tune-yyz/encode_corpus/output/llama3_shu/"
os.makedirs(output_dir, exist_ok=True)  # 如果输出目录不存在，则创建它

# 保存洗牌后的数据到新的.pkl文件
file_idx = 0
file_name_base = os.path.join(output_dir, "shuffled_llama3")
while file_idx < len(data_pairs):
    # 创建新的文件名
    output_file_name = f"{file_name_base}_{file_idx}.pkl"
    
    # 打开文件并写入洗牌后的数据
    with open(output_file_name, "wb") as f:
        for _ in range(10000):  # 假设args.conversations_per_file定义了每个文件保存多少个问答对
            if file_idx >= len(data_pairs):
                break
            pickle.dump(data_pairs[file_idx], f)
            file_idx += 1
    txt_fn = output_file_name[:-len(".pkl")] + '_len.txt'
    with open(txt_fn, "w") as f:
            print(len(data_pairs), file=f)
print(f"Shuffled data has been saved to {output_dir}")