import json
import pickle
import glob
import random

"""
数据集格式:
data_0_len.txt 第0个文件的数据集长度
data_0.pkl     
data_1_len.txt 第1个文件的数据集长度
data_1.pkl
data_2_len.txt 第2个文件的数据集长度
data_2.pkl
...

保存pkl文件时, 每条数据都需要pickle.dump一次, 读取时, 多次pickle.load直到EOF
"""

def wizard_file_iter(fn):
    with open(fn, "rb") as f:
        while True:
            try:
                input_ids, labels = pickle.load(f)
                yield {
                    "input_ids": input_ids,
                    "labels": labels
                }
            except EOFError:
                break

def llama2func(fn):
    with open(fn, "r") as f:
        datas = json.load(f) 
    # 构建包含原始问题和初始回答的上下文
    now = 0
    print(len(datas))
    for data in datas:
        question = data["instruction"]
        answer = data["output"]
        if now < 3:
            print(len(datas))
            print(f"question = {question}")
            print(f"correct_answer = {answer}")
            now += 1
        yield {
            "question": question,
            "answer": answer
        }
        
PRETRAIN_CORPUS = {
    # "pile": {
    #     "file_list": glob.glob("./dataset/train_once/pile/*.pkl"),
    #     "file_iter": pile_file_iter,
    # },
    # "openhermes": {
    #     "file_list": glob.glob("./dataset/train_once/openhermes/*.pkl"),
    #     "file_iter": openhermes_file_iter,
    # },
    # "wizard": {
    #     "file_list": glob.glob("./dataset/train_once/openhermes/*.pkl"),
    #     "file_iter": wizard_file_iter,
    # }
    # "wizard": {
    #     "file_list": glob.glob("/data1/dcy/refine-llama2-7b-20240315/yyz/encode_corpus/output/wizard/*.pkl"),
    #     "file_iter": wizard_file_iter,
    # }
    #  "wizard": {
    #     "file_list": glob.glob("/data1/dcy/projects/fine-tune/fine-tune-yyz/encode_corpus/output/wizard_llama3/*.pkl"),
    #     "file_iter": wizard_file_iter,
    # }
    # "llama2-sva": {
    #     "file_list": glob.glob("/data1/dcy/projects/fine-tune/fine-tune-yyz/encode_corpus/output/llama2-sva/llama2-sva_0.pkl"),
    #     "file_iter": wizard_file_iter,
    # },
    # "llama2-gsm": {
    #     "file_list": glob.glob("/data1/dcy/projects/fine-tune/fine-tune-yyz/encode_corpus/output/llama2-gsm/llama2-gsm_0.pkl"),
    #     "file_iter": wizard_file_iter,
    # },
    # "llama2-gsm": {
    #     "file_list": glob.glob("/data1/dcy/projects/fine-tune/fine-tune-yyz/encode_corpus/output/llama2-gsm/llama2-gsm_1.pkl"),
    #     "file_iter": wizard_file_iter,
    # },
    # "llama3":{
    #     "file_list": glob.glob("/data1/dcy/projects/fine-tune/fine-tune-yyz/encode_corpus/output/llama3/llama3_0.pkl"),
    #     "file_iter": wizard_file_iter,
    # },
    # "vicuna-7b":{
    #     "file_list": glob.glob("/data1/dcy/projects/fine-tune/fine-tune-yyz/encode_corpus/output/vicuna-7b/vicuna-7b_0.pkl"),
    #     "file_iter": wizard_file_iter,
    # },
    # "vicuna-13b":{
    #     "file_list": glob.glob("/data1/dcy/projects/fine-tune/fine-tune-yyz/encode_corpus/output/vicuna-13b/vicuna-13b_0.pkl"),
    #     "file_iter": wizard_file_iter,
    # },
    # "vicuna-33b":{
    #     "file_list": glob.glob("/data1/dcy/projects/fine-tune/fine-tune-yyz/encode_corpus/output/vicuna-33b/vicuna-33b_0.pkl"),
    #     "file_iter": wizard_file_iter,
    # },
    # "qwen_7b":{
    #     "file_list": glob.glob("/data1/dcy/projects/fine-tune/fine-tune-yyz/encode_corpus/output/Qwen_7B/Qwen_7B_0.pkl"),
    #     "file_iter": wizard_file_iter,
    # },
    # "qwen_14b":{
    #     "file_list": glob.glob("/data1/dcy/projects/fine-tune/fine-tune-yyz/encode_corpus/output/Qwen_14B/Qwen_14B_0.pkl"),
    #     "file_iter": wizard_file_iter,
    # },    
    #     "qwen1_5_14b":{
    #     "file_list": glob.glob("/data1/dcy/projects/fine-tune/fine-tune-yyz/encode_corpus/output/Qwen1_5_14B/Qwen1_5_14B_0.pkl"),
    #     "file_iter": wizard_file_iter,
    # },    
    # "llama2_13b":{
    #     "file_list": glob.glob("/data1/dcy/projects/fine-tune/fine-tune-yyz/encode_corpus/output/llama2_13b/llama2_13b_0.pkl"),
    #     "file_iter": wizard_file_iter,
    # },   
    # "llama2_13b_70b":{
    #     "file_list": glob.glob("/data1/dcy/projects/fine-tune/fine-tune-yyz/encode_corpus/output/llama2_13b_70b/llama2_13b_70b_0.pkl"),
    #     "file_iter": wizard_file_iter,
    # },  
    "llama3_shuffle":{
        "file_list": glob.glob("/data1/dcy/projects/fine-tune/fine-tune-yyz/encode_corpus/output/llama3_shu/*.pkl"),
        "file_iter": wizard_file_iter,
    },    
#     "llama3_shuffle":{
#         "file_list": glob.glob("/data1/dcy/projects/fine-tune/fine-tune-yyz/encode_corpus/output/llama3_default/llama3_default_0.pkl"),
#         "file_iter": wizard_file_iter,
#     },    
}


def get_dataset_generator(args,shuffle=True, ):
    """
        获得数据集生成器, 生成器能生成整个数据集
        shuffle (bool): 是否随机顺序生成
    """
    generators = []
    for corpus, d in PRETRAIN_CORPUS.items():
        for fn in d["file_list"]:
            generators.append(d["file_iter"](fn))
    if shuffle:  # 随机从某个文件采样
        active_generators_indices = list(range(len(generators)))  # 还有数据的生成器
        while True:
            if not active_generators_indices:  # 所有生成器均生成完毕
                break
            idx = random.choice(active_generators_indices)  # 随机选择一个生成器
            try:
                data = next(generators[idx])  # 从随机选择的生成器中获得一个元素
                yield data
            except StopIteration:  # 该生成器已经没有数据了，从活跃列表中移除
                active_generators_indices.remove(idx)
    else:  # 按数据集和文件顺序采样
        for gen in generators:
            for data in gen:
                yield data


def get_dataset_len():
    #  根据data_i_len.txt文件计算整个数据集的长度
    sum = 0
    for corpus, d in PRETRAIN_CORPUS.items():
        # print(corpus)
        for file_name in d["file_list"]:
            fn = file_name[:-len(".pkl")]+"_len.txt"
            with open(fn, "r") as f:
                sum += int(f.readline())
    return sum


if __name__ == "__main__":
    print(get_dataset_len())
    gen = get_dataset_generator()
    with open("1.txt", "w") as f:
        for data in gen:
            print(data, file=f)
    