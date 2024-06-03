import random
from dataset_generator import get_dataset_len, get_dataset_generator
import numpy as np
import torch
from torch.utils.data import Dataset
import pickle

class GeneratorDataset(Dataset):
    """
        生成器数据集, 数据集由生成器得到, 需要修改dataset_generator.py来获取不同数据集
    """
    def __init__(self, args):
        self.gen = get_dataset_generator(args, shuffle=True,)
        self.len = get_dataset_len()

    def __len__(self):
        return self.len

    def __getitem__(self, args):
        try:
            data = next(self.gen)
        except StopIteration:
            self.gen = get_dataset_generator(args,shuffle=True,)
            data = next(self.gen)
        input_ids = np.array(data["input_ids"], dtype=np.int32)
        labels = np.array(data["labels"], dtype=np.int32)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long, device=torch.device("cuda")),
            "labels": torch.tensor(labels, dtype=torch.long, device=torch.device("cuda"))
        }

class DIRDataset(Dataset):
    """
    生成器数据集, 数据集由生成器得到, 在构造函数中一次性加载所有数据
    """
    def __init__(self):
        # 获取生成器
        gen = get_dataset_generator(shuffle=True)
        # 读取所有数据DIR
        self.data = []
        for _ in range(get_dataset_len()):
            data = next(gen, None)
            if data is None:
                break
            self.data.append(data)
        
        # 转换数据格式
        self.data = [{
            "input_ids": np.array(d["input_ids"], dtype=np.int32),
            "labels": np.array(d["labels"], dtype=np.int32)
        } for d in self.data]
        # 如果需要，打乱数据
        random.shuffle(self.data)
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 从预加载的数据中获取样本
        sample = self.data[idx]
        return {
            "input_ids": torch.tensor(sample["input_ids"], dtype=torch.long, device=torch.device("cuda")),
            "labels": torch.tensor(sample["labels"], dtype=torch.long, device=torch.device("cuda"))
        }

class MyDataset(Dataset):
    """
        普通dataset, 根据file_path直接读入所有数据
    """
    def __init__(self, data_path):
        #  该方法需要根据实际情况修改
        self.data = []  # list[(input_ids, labels)]
    
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return {
            "input_ids": torch.tensor(self.data[i][0], dtype=torch.long, device=torch.device("cuda")),
            "labels": torch.tensor(self.data[i][1], dtype=torch.long, device=torch.device("cuda"))
        }

class DatasetIds(Dataset):
    '''
    Dataset construction for training GPT-2 model, without padding. Truncation is done using the end-of-sequence (EOS) token.
    This dataset directly loads preprocessed data, eliminating the need for waiting.
    '''
    
    def __init__(self):
        
        super().__init__()
        datas = pickle.load(open("/data1/dcy/projects/fine-tune/fine-tune-yyz/encode_corpus/output/llama2-sva/llama2-sva_0.pkl", "rb"))
        self.input_ids = datas['input_ids']
        print(f"{len(self.input_ids)}")
        max_length = 2048
        self.attention_mask = datas['attention_mask']
        self.labels = datas['labels']
        self.max_length = max_length

    def __len__(self):
        return len(self.input_ids) // self.max_length

    def __getitem__(self, index):
        return torch.tensor(self.input_ids[index * self.max_length : (index + 1) * self.max_length]), \
                torch.tensor(self.labels[index * self.max_length : (index + 1) * self.max_length]), \
                    torch.tensor(self.attention_mask[index * self.max_length : (index + 1) * self.max_length])
if __name__ == '__main__':
    data = DatasetIds()
  