import multiprocessing
import time
import pickle
import argparse
import json
import glob
import os
import numpy as np
from transformers import AutoTokenizer
import os

def openhermes_func(file_name):
    with open(file_name, "r") as f:
        print("loading openhermes")
        lst = json.load(f)  # 1001551 lines
        for dct in lst:
            d = dct["conversations"]
            q, a = d[0]["value"], d[1]["value"]
            yield {
                "question": q,
                "answer": a
            }


OVERALL_INSTRUCTION = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."

def format_dialogue(query, history):
    prompt = ""

    # 添加历史对话部分
    for i, (old_query, response) in enumerate(history):
        if i == 0:  # 首个历史条目，包含总体指导说明
            prompt += f"<s>[INST] <<SYS>>\n\n{OVERALL_INSTRUCTION}\n\n<</SYS>>\n\n{old_query} [/INST] {response} </s>"
        else:  # 后续历史条目
            prompt += f"<s>[INST] {old_query} [/INST] {response} </s>"

    # 添加当前查询部分
    prompt += f"<s>[INST] {query} [/INST]"

    return prompt
def wizard_func(fn):
    refine_query = "Please further think about and give me a more precise and professional answer.\n" 

    with open(fn, "r") as f:
        datas = json.load(f) 
    now = 1
    for data in datas:
        # 构建包含原始问题和初始回答的上下文
        question = format_dialogue(refine_query, history=[(data["question"], data["output1"])])

        correct_answer = data["output2"]
        if now == 1:
            print(len(datas))
            print(f"question = {question}")
            print(f"correct_answer = {correct_answer}")
            now += 1
        yield {
            "question": question,
            "answer": correct_answer
        }

from typing import (
    AbstractSet,
    cast,
    Collection,
    Dict,
    Iterator,
    List,
    Literal,
    Sequence,
    TypedDict,
    Union,
)

Role = Literal["user", "assistant"]


class Message(TypedDict):
    role: Role
    content: str


Dialog = Sequence[Message]

class ChatFormat:
    def __init__(self):
        self.input = ""

    def add_header(self, message: Message):
        tokens = []
        tokens.append("<|start_header_id|>")
        tokens.extend(message["role"])
        tokens.append("<|end_header_id|>")
        tokens.extend("\n\n")
        return tokens

    def add_message(self, message: Message):
        tokens = self.add_header(message)
        tokens.extend(message["content"].strip())
        tokens.append("<|eot_id|>")
        return tokens

    def add_dialog_prompt(self, dialog: Dialog):
        tokens = []
        tokens.append("<|begin_of_text|>")
        for message in dialog:
            tokens.extend(self.add_message(message))
        # Add the start of an assistant message for the model to complete.
        tokens.extend(self.add_header({"role": "assistant", "content": ""}))
        return "".join(tokens) 



def wizard_func_llama3(fn):
  
    with open(fn, "r") as f:
        datas = json.load(f) 
    now = 1
    refine_query = "Please further think about and give me a more precise and professional answer.\n" 

    for data in datas:
        format = ChatFormat()
        dialog = [
            {
                "role": "user",
                "content": data["question"],
            },
            {
                "role": "assistant",
                "content": data["output1"],
            },
            {
                "role": "user",
                "content": refine_query,
            },
        ]

        question = format.add_dialog_prompt(dialog)
        # 构建包含原始问题和初始回答的上下文
        correct_answer = data["output2"]
        if now < 3:
            print(len(datas))
            print(f"question = {question}")
            print(f"correct_answer = {correct_answer}")
            now += 1
            print(f"\n\n\n\n")
        yield {
            "question": question,
            "answer": correct_answer
        }


def base_func_llama3(fn):
  
    with open(fn, "r") as f:
        datas = json.load(f) 
    now = 1
    for data in datas:
        
        question = data["question"]
        correct_answer = data["output1"] 
        if now < 3:
            print(len(datas))
            print(f"question = {question}")
            print(f"correct_answer = {correct_answer}")
            now += 1
            print(f"\n\n")
        yield {
            "question": question,
            "answer": correct_answer
        }
def format_prompt_default(query, history):
    HUMAN_BEGIN = ""
    ASSISTANT_BEGIN = ""
    prompt = ""
    for old_query, response in history:
        prompt += HUMAN_BEGIN + old_query
        prompt += ASSISTANT_BEGIN + response
        prompt += "\n"
    prompt += HUMAN_BEGIN + query + ASSISTANT_BEGIN
    return prompt
def refine_func_llama3(fn):
  
    with open(fn, "r") as f:
        datas = json.load(f) 
    now = 1
    refine_query = "Please further think about and give me a more precise and professional answer." 

    for data in datas:
        
        question = data["question"] + data["output1"] + "\n" + "<|end_of_text|>"+ refine_query
        correct_answer = data["output2"] 
        if now < 3:
            print(len(datas))
            print(f"question = {question}")
            print(f"correct_answer = {correct_answer}")
            now += 1
            print(f"\n\n")
        yield {
            "question": question,
            "answer": correct_answer
        }



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

class VicunaChatFormat:
    def __init__(self):
        self.input = ""

    def add_header(self, message: Message,i = 1):
        tokens = []
        begin = ["<s>", " "]
        tokens.extend(str(begin[i % 2]))
        tokens.extend(message["role"])
        tokens.extend(": ")
        return tokens

    def add_message(self, message: Message, i):
        
        seps = [" ", "</s>"]
        
        tokens = self.add_header(message, i)
        
        tokens.extend(message["content"].strip())
        tokens.extend(str(seps[i % 2]))
        return tokens

    def add_dialog_prompt(self, dialog: Dialog):
        tokens = []
        for i, message in enumerate(dialog):
            tokens.extend(self.add_message(message, i))
        # Add the start of an assistant message for the model to complete.
        tokens.extend(self.add_header({"role": "ASSISTANT", "content": ""}))
        return "".join(tokens)  # Return a single string concatenated from the list of tokens.


def wizard_func_vicuna(fn):
  
    with open(fn, "r") as f:
        datas = json.load(f) 
    now = 1
    refine_query = "Please further think about and give me a more precise and professional answer.\n" 
    for data in datas:
        dialog = [
            {
                "role": "USER",
                "content": data["question"],
            },
            {
                "role": "ASSISTANT",
                "content": data["output1"],
            },
            {
                "role": "USER",
                "content": refine_query,
            },
        ]
        format = VicunaChatFormat()
        question = format.add_dialog_prompt(dialog)
        # 构建包含原始问题和初始回答的上下文
        correct_answer = data["output2"]
        if now < 3:
            print(len(datas))
            print(f"question = {question}")
            print(f"correct_answer = {correct_answer}")
            now += 1
        yield {
            "question": question,
            "answer": correct_answer
        }
def wizard_func_qwen(fn):
  
    with open(fn, "r") as f:
        datas = json.load(f) 
    now = 1
    refine_query = "Please further think about and give me a more precise and professional answer.\n" 

    for data in datas:
        question = f'Q: {data["question"]} \n\nA: {data["output1"]} \n\nQ: {refine_query} \n\nA:'
        # 构建包含原始问题和初始回答的上下文
        correct_answer = data["output2"]
        if now < 3:
            print(len(datas))
            print(f"question = {question}")
            print(f"correct_answer = {correct_answer}")
            now += 1
            print(f"\n\n\n\n")
        yield {
            "question": question,
            "answer": correct_answer
        }

class Qwen1_5ChatFormat:
    def __init__(self):
        self.input = ""

    def add_header(self, message: Message):
        tokens = []
        tokens.append("<|im_start|>")
        tokens.extend(message["role"])
        tokens.extend("\n")
        return tokens

    def add_message(self, message: Message):
        tokens = self.add_header(message)
        tokens.extend(message["content"].strip())
        tokens.append("<|im_end|>\n")
        return tokens

    def add_dialog_prompt(self, dialog: Dialog):
        tokens = []
        for message in dialog:
            tokens.extend(self.add_message(message))
        # Add the start of an assistant message for the model to complete.
        tokens.extend(self.add_header({"role": "assistant", "content": ""}))
        return "".join(tokens)  # Return a single string concatenated from the list of tokens.




def wizard_func_qwen1_5(fn):
  
    with open(fn, "r") as f:
        datas = json.load(f) 
    now = 1
    
    refine_query = "Please further think about and give me a more precise and professional answer.\n" 
    for data in datas:
        dialog = [
            {
                "role": "user",
                "content": data["question"],
            },
            {
                "role": "assistant",
                "content": data["output1"],
            },
            {
                "role": "user",
                "content": refine_query,
            },
        ]
        format = Qwen1_5ChatFormat()
        question = format.add_dialog_prompt(dialog)
        # 构建包含原始问题和初始回答的上下文
        correct_answer = data["output2"]
        if now < 3:
            print(len(datas))
            print(f"question = {question}")
            print(f"correct_answer = {correct_answer}")
            now += 1
        yield {
            "question": question,
            "answer": correct_answer
        }        
        
        
def llama3_default(fn):
  
    with open(fn, "r") as f:
        datas = json.load(f) 
    now = 1
    
    refine_query = "Please further think about and give me a more precise and professional answer.\n" 
    for data in datas:
        question = data["question"]+data["output1"]+ "<|end_of_text|>"+refine_query 
        # 构建包含原始问题和初始回答的上下文
        correct_answer = data["output2"]
        if now < 3:
            print(len(datas))
            print(f"question = {question}")
            print(f"correct_answer = {correct_answer}")
            now += 1
        yield {
            "question": question,
            "answer": correct_answer
        }        
        
        
CORPUS = {
    # "example": {
    #     "file_list": glob.glob("./*.jsonl"),  # 该语料库的文件列表
    #     "func": None  # 读取该语料库单个文件的函数
    # },
    # "wizard": {
    #     "file_list": glob.glob("/data1/dcy/refine-llama2-7b-20240315/code_example/train/data/llama2_7b_20230315/llama2_qwen14b_output_data_file.pkl"),
    #     "func": wizard_func,
    # }
    # "wizard": {
    #     "file_list": glob.glob("/data1/dcy/refine-llama2-7b-20240315/yyz/train/merged_data.json"),
    #     "func": wizard_func,
    # }
    # "wizard_llama3": {
    #     "file_list": glob.glob("/data1/dcy/projects/fine-tune/fine-tune-yyz/train/merged_data.json"),
    #     "func": wizard_func_llama3,
    # }
    # "llama2-gsm": {
    #     "file_list": glob.glob("/data1/dcy/projects/ziqin/mmlu-few-shot/myresult/zqin/gsm.json"),
    #     "func": llama2func,
    # },
    # "llama2-sva": {
    #     "file_list": glob.glob("/data1/dcy/projects/ziqin/mmlu-few-shot/myresult/zqin/sva.json"),
    #     "func": llama2func,
    # }
    # "vicuna-7b": {
    #     "file_list": glob.glob("/data1/dcy/projects/fine-tune/fine-tune-yyz/encode_corpus/data/vicuna-7b/vicuna7b_merge.json"),
    #     "func": wizard_func_vicuna,
    # }, 
    # "vicuna-13b": {
    #     "file_list": glob.glob("/data1/dcy/projects/fine-tune/fine-tune-yyz/encode_corpus/data/vicuna-13b/vicuna13b_merge.json"),
    #     "func": wizard_func_vicuna,
    # }, 
    # "vicuna-33b": {
    #     "file_list": glob.glob("/data1/dcy/projects/fine-tune/fine-tune-yyz/encode_corpus/data/vicuna-33b/vicuna33b_merge.json"),
    #     "func": wizard_func_vicuna,
    # }, 
    # "Qwen_14B": {
    #     "file_list": glob.glob("/data1/dcy/projects/fine-tune/fine-tune-yyz/encode_corpus/data/Qwen_14B/Qwen_14B_merge.json"),
    #     "func": wizard_func_qwen,
    # },
    # "Qwen_7B": {
    #     "file_list": glob.glob("/data1/dcy/projects/fine-tune/fine-tune-yyz/encode_corpus/data/Qwen_7B/Qwen_7B_merge.json"),
    #     "func": wizard_func_qwen,
    # },
    # "Qwen1_5_14B": {
    #     "file_list": glob.glob("/data1/dcy/projects/fine-tune/fine-tune-yyz/encode_corpus/data/Qwen_14B/Qwen_14B_merge.json"),
    #     "func": wizard_func_qwen1_5,
    # },
    # "llama2_13b":{
    #     "file_list": glob.glob("/data1/dcy/projects/fine-tune/fine-tune-yyz/encode_corpus/data/llama2_13b/llama2_13b_merge.json"),
    #     "func":wizard_func
    # },
    # "llama2_13b_70b":{
    #     "file_list": glob.glob("/data1/dcy/projects/fine-tune/fine-tune-yyz/encode_corpus/data/llama2_13b_70b/llama2_13b_70b_merge.json"),
    #     "func":wizard_func
    # }
    # "llama3": {
    #     "file_list": glob.glob("/data1/dcy/projects/fine-tune/fine-tune-yyz/encode_corpus/data/llama3/llama3_merge.json"),
    #     "func": wizard_func_llama3,
    # }, 
    # "llama3_base": {
    #     "file_list": glob.glob("/data1/dcy/projects/fine-tune/fine-tune-yyz/encode_corpus/data/llama3_all/*.json"),
    #     "func": base_func_llama3,
    # }, 
    "llama3_alll": {
        "file_list": glob.glob("/data1/dcy/projects/fine-tune/fine-tune-yyz/encode_corpus/data/llama3_all/*.json"),
        "func": wizard_func_llama3,
    }, 
    # "llama3_invove":{
    #     "file_list": glob.glob("/data1/dcy/projects/fine-tune/fine-tune-yyz/encode_corpus/data/my_merge_data_llama2_merge.json"),
    #     "func": wizard_func_llama3,
    # }
}
# refine_func_llama3 base_func_llama3
def merge_datas(datas, max_length, dtype, pad_token_id=128001):
    """
        将datas的部分数据合并, 并padding到max_length
    """
    cur_input_ids = []
    cur_labels = []
    new_datas = []
    for (input_ids, labels) in datas:
        if len(cur_input_ids) + len(input_ids) <= max_length:
            cur_input_ids += input_ids
            cur_labels += labels
        else:
            pad_len = max_length - len(cur_input_ids)
            cur_input_ids += [pad_token_id] * pad_len
            cur_labels += [-100] * pad_len
            new_datas.append([cur_input_ids, cur_labels])
            cur_input_ids, cur_labels = input_ids, labels
    
    pad_len = max_length - len(cur_input_ids)
    cur_input_ids += [pad_token_id] * pad_len
    cur_labels += [-100] * pad_len
    new_datas.append([cur_input_ids, cur_labels])
    
    new_datas = np.array(new_datas, dtype=dtype)
    return new_datas
            

# def pad_datas(datas, max_length, dtype, pad_token_id=128001):
#     new_datas = []
#     for (input_ids, labels) in datas:
#         if len(input_ids) <= max_length:
#             pad_len = max_length - len(input_ids)
#             input_ids += [pad_token_id] * pad_len
#             labels += [-100] * pad_len
#             new_datas.append([input_ids, labels])
#     new_datas = np.array(new_datas, dtype=dtype)
#     return new_datas


def save_file(datas, file_name, args, dtype=np.int32):
    """
        将input_ids_list存入文件
        input_ids_list (list[int]): input_ids的数组
        labels_list (list[int]): labels的数组
        file_name (str): 保存文件名
        generator_mode (bool): 见parser中的help
    """
    datas = merge_datas(datas, args.max_length, dtype, args.pad_token_id)
    # datas = pad_datas(datas, args.max_length, dtype, args.pad_token_id)
    if args.generator_mode:
        with open(file_name, "wb") as f:
            for data in datas:
                pickle.dump(data, f)
        txt_fn = file_name[:-len(".pkl")] + '_len.txt'
        with open(txt_fn, "w") as f:
            print(len(datas), file=f)
    else:
        with open(file_name, "wb") as f:
            pickle.dump(datas, f)


def process_initializer(tokenizer_path):
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    # tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True,eos_token = '<|endoftext|>', pad_token='<|endoftext|>')
    
    st = "<|end_of_text|>"
    if st not in tokenizer.additional_special_tokens:
            tokenizer.add_special_tokens({'additional_special_tokens': tokenizer.additional_special_tokens + [st]})
            # print('additional_special_tokens:', tokenizer.additional_special_tokens)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    args.pad_token_id = tokenizer.eos_token_id
    # print(f"'eos_token:'={tokenizer.eos_token}")
    # print(f"'eos_token_id:'={tokenizer.eos_token_id}")
    # print(f"'pad_token:'={tokenizer.pad_token}")
    # print(f"'pad_token_id:'={tokenizer.pad_token_id}")

def add(conversation:dict):
    """ 
        将对话预处理为input_ids和labels, 其中input_ids=[q][a][pad], labels=[-100][a][-100]
        conversation (dict): 问题和回答的原文本, 包含"question"和"answer"两个key
        return: 处理后得到的数据集(input_ids, labels), 不包含padding的有效token数量
    """
    q_ids = tokenizer(conversation["question"])["input_ids"]
    a_ids = tokenizer(conversation["answer"])["input_ids"]
    input_ids = q_ids + a_ids
    labels = [-100] * len(q_ids) + a_ids
    real_len = len(input_ids)  # 非padding部分的token数量
    if real_len < args.max_length:
        input_ids += [tokenizer.eos_token_id] * (args.max_length - real_len)
        labels += [tokenizer.eos_token_id] + [-100] * (args.max_length - real_len - 1)
    return input_ids, labels, real_len

def encode(conversation:dict):
    """ 
        将对话预处理为input_ids和labels, 其中input_ids=[q][a][pad], labels=[-100][a][-100]
        conversation (dict): 问题和回答的原文本, 包含"question"和"answer"两个key
        return: 处理后得到的数据集(input_ids, labels)
    """
    q_ids = tokenizer(conversation["question"])["input_ids"]
    a_ids = tokenizer(conversation["answer"])["input_ids"]
    if a_ids[0] == tokenizer.bos_token_id:
        a_ids = a_ids[1:]
    input_ids = q_ids + a_ids + [tokenizer.eos_token_id]
    labels = [-100] * len(q_ids) + a_ids + [tokenizer.eos_token_id]
    return input_ids, labels

def encode_corpus(corpus):
    """
        对名为corpus的语料库进行编码, 在全局变量CORPUS中找到其定义
        corpus (str): 语料库名称
    """
    print(f"processing {corpus}")
    pool = multiprocessing.Pool(args.process_cnt, initializer=process_initializer, initargs=(args.tokenizer_path,))
    path = os.path.join(args.output_dir, corpus)
    os.makedirs(path, exist_ok=True)
    file_idx = 0  # 输出文件编号
    start_time = time.time()
    datas = []
    total_token_processed = 0  # 总共处理了多少个tokens
    cur_token_processes = 0  # 当前处理了多少个tokens
    for file_name in CORPUS[corpus]["file_list"]:
        file_iter = CORPUS[corpus]["func"](file_name)  # jsonl文件迭代器 
        encoded_docs_iter = pool.imap_unordered(encode, file_iter, args.process_cnt)

        for i, (input_ids, labels) in enumerate(encoded_docs_iter):
            if len(input_ids) > args.max_length:
                continue  # 忽略token数量大于模型输入长度的问答

            cur_token_processes += len(input_ids)
            total_token_processed += len(input_ids)
            datas.append((input_ids, labels))

            if i % args.print_step == 0:  # 每处理print_step行文本需要输出一下运行效率, 便于观察
                elapsed = time.time() - start_time
                second = int(elapsed)
                print(f"\rprocessed {total_token_processed:.2e} tokens, run for {(second // 3600):02d}:{(second // 60 % 60):02d}:{(second % 60):02d}, {(total_token_processed / elapsed):.2e} tokens/s", end="")

            if cur_token_processes >= args.tokens_per_file:  # 每个文件保存conversations_per_file个问答
                fn = f"{args.output_dir}/{corpus}/{corpus}_{file_idx}.pkl"
                save_file(datas, fn, args)
                file_idx += 1
                datas = []
                cur_token_processes = 0

        if cur_token_processes > 0:
            fn = f"{args.output_dir}/{corpus}/{corpus}_{file_idx}.pkl"
            save_file(datas, fn, args)

    print(f"\nprocessed {total_token_processed} tokens.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_path", type=str)
    parser.add_argument("--process_cnt", type=int, default=1)
    parser.add_argument("--max_length", type=int)
    parser.add_argument("--generator_mode", action="store_true", help="若使用generator_mode, 则pkl文件需要多次读取, 每次读取一个input_ids; 若不使用generator_mode, 则pkl文件一次读取所有input_ids")
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--corpus", nargs="+")
    parser.add_argument("--tokens_per_file", type=int, default=1e9)
    parser.add_argument("--print_step", type=int, default=100)
    parser.add_argument("--pad_token_id", type=int, default=128001)
    args = parser.parse_args()
    print("args =", args)

    for corpus in args.corpus:
        print(f"corpus ={corpus}" )
        encode_corpus(corpus)
