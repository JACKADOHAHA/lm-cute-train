import argparse, os, json, random, datetime
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import json, os
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import deepspeed
from peft import get_peft_model, PeftModel

from config import lora_config, DS_CONFIG_lora, DS_CONFIG_ft
from dataset import DIRDataset,GeneratorDataset
def get_model_layers(model):
    layers = [["", model]]
    i = 0
    while i < len(layers):
        for nc, lc in layers[i][1].named_children():
            layers.append([f"{layers[i][0]}.{nc}" if layers[i][0] else nc, lc])
        i += 1
    return layers


def print0(*args, **kwargs):
    """只在主进程print"""
    if torch.distributed.get_rank() == 0:
        print(*args, **kwargs)


def setup_distributed_environment(local_rank):
    """配置分布式训练环境"""
    if local_rank != -1:  # 使用分布式训练
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",  # 使用环境变量指定初始化方法。这意味着PyTorch会自动从环境变量中寻找必要的设置，如主机地址和端口号，以及进程的排名和总数。
            rank=local_rank,  # 设置当前进程的排名。
            world_size=torch.cuda.device_count(),  # 设置进程组中的进程总数，这里使用的是当前节点上可用的CUDA设备数
        )
    else:  # 单卡训练
        device = torch.device("cuda")
    deepspeed.init_distributed()
    return device


def initialize_model(args, device):
    """加载和初始化模型"""
    print0("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, pad_token='<|endoftext|>')
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        # attn_implementation="flash_attention_2",
        low_cpu_mem_usage=False,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    ).to(device)

    if args.use_lora:
        if args.load_lora:  # 加载lora存档点
            print0(f"Loading LoRa checkpoint from {args.load_lora_path}")
            model = PeftModel.from_pretrained(model, args.load_lora_path, is_trainable=True)
        else:  # 从base开始训练
            print0("Using LoRa: Training from scratch")
            model = get_peft_model(model, lora_config)
    elif args.add_token:
        add_tokens = [
            "<s>",
            "</s>"
            "<|endoftext|>"
            "<|begin_of_text|>",
            "<|end_of_text|>",
            "<|start_header_id|>",
            "<|end_header_id|>",
            "<|eot_id|>",  # end of turn
        ] 
        num_added_tokens = tokenizer.add_special_tokens({'additional_special_tokens': add_tokens})
        model.resize_token_embeddings(len(tokenizer))  # 调整 embed 层,使其能够适应新的 token 数量
        embedding_layer = model.get_input_embeddings()  # 获取 embedding 层
        with torch.no_grad():
            new_token_indices = range(len(tokenizer) - num_added_tokens, len(tokenizer))
            for token_index in new_token_indices:
                embedding_layer.weight[token_index].uniform_(-0.1, 0.1)  # 均匀分布初始化
    return model, tokenizer


def prepare_dataloader(args, deepspeed_config):
    """准备数据加载器"""
    print0("Loading dataset...")

    train_dataset = GeneratorDataset(args)
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if args.local_rank != -1 else None
    train_dataloader = DataLoader(
        dataset=train_dataset,
        sampler=train_sampler,
        batch_size=deepspeed_config["train_micro_batch_size_per_gpu"],
        num_workers=0,
    )
    return train_dataloader


def train_model(model, device, tokenizer, train_dataloader, ds_config, args, save_dir):
    """
        模型训练循环
    """
    engine, _, _, _ = deepspeed.initialize(
        config=ds_config,
        model=model,
        model_parameters=model.parameters(),
    )
    step = 0
    losses = []
    if dist.get_rank() == 0:
        pbar = tqdm(total=args.max_steps, ncols=95)
    
    for epoch in range(args.max_epochs):
        for batch_id, batch in enumerate(train_dataloader):
            # print (f"batch['input_ids'] = {batch['input_ids'].max()}")
            # print (f"batch['labels'] = {batch['labels'].max()}")
            loss = engine(
                input_ids=batch["input_ids"].to(device),
                labels=batch["labels"].to(device),
                use_cache=False
            ).loss
            # loss = engine(  # 前向传播
            #     input_ids=batch[0].to(device),
            #     labels=batch[1].to(device),
            #     attention_mask=batch[2].to(device),
            #     use_cache=False
            # ).loss
            engine.backward(loss)
            engine.step()
            step += 1
            losses.append(loss.item())
            if dist.get_rank() == 0:
                pbar.update()
                pbar.set_description(f"epoch:{epoch + 1},batch:{batch_id + 1}/{len(train_dataloader)},loss:{np.mean(losses[-200:]):.4f}")

            if step % args.save_steps == 0:
                save_checkpoint(engine, tokenizer, args, step, losses, save_dir)
            
            if step >= args.max_steps:
                break
        if step >= args.max_steps:
            break
    
    if args.max_steps % args.save_steps != 0:
        save_checkpoint(engine, tokenizer, args, step, losses, save_dir)

    if dist.get_rank() == 0:
        pbar.close()


def save_checkpoint(engine, tokenizer, args, step, losses, save_dir):
    """
        保存模型和训练损失
    """
    ckpt_dir = os.path.join(save_dir, args.ckpt_dir, f"{args.save_name}_{step}")
    os.makedirs(ckpt_dir, exist_ok=True)

    # 保存模型
    if args.use_lora:
        if args.local_rank != -1:  # 是分布式训练环境
            dist.barrier()  # 阻塞当前进程, 直到所有其他进程也调用了 dist.barrier(), 才会释放所有进程
        if torch.distributed.get_rank() == 0 or args.local_rank == -1:  # 主进程或非分布式训练环境
            engine.save_pretrained(ckpt_dir)  
        if args.local_rank != -1:
            dist.barrier()
    else:
        engine.save_16bit_model(ckpt_dir)  # 保存模型
        with open(os.path.join(ckpt_dir, 'config.json'), 'w') as f:  # 保存config
            print(json.dumps(engine.module.config.to_dict(), indent=4), file=f)
        tokenizer.save_pretrained(ckpt_dir)  # 保存tokenizer
    
    # 保存损失函数
    loss_file_name = os.path.join(ckpt_dir, "loss_list.json")
    with open(loss_file_name, "w") as f:
        print(json.dumps(losses), file=f)


    # 读取loss_list
    loss_fn = os.path.join(ckpt_dir, "loss_list.json")
    with open(loss_fn, "r") as f:
        l = json.load(f)

    x, y = [], []
    n_points = 100  # 图上保留几个点
    step = len(l) // n_points 
    for i in range(0, len(l), step):
        x.append(i + 1)
        y.append(np.mean(l[i:i + step]))  # 取一段loss的平均值，而非单点取值
    plt.cla()
    # 绘制loss图片
    plt.title("train loss")
    plt.xlabel("steps")
    plt.ylabel("loss")
    plt.plot(x, y)

    # 将loss图片保存在对应checkpoint文件夹中
    fn = os.path.join(ckpt_dir, ckpt_dir.split("/")[-1] + "_loss.png")
    print("save loss picture at:", fn)
    plt.savefig(fn)

def set_seed(seed):
    """设置随机数种子, 保证结果可重现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def initialize_config(args):
    set_seed(args.seed)
    ds_config = DS_CONFIG_lora if args.use_lora else DS_CONFIG_ft
    t = datetime.datetime.now()
    save_dir = f"{args.output_dir}/{t.month}-{t.day}_{t.hour:02d}:{t.minute:02d}_{args.save_name}"
    os.makedirs(save_dir, exist_ok=True)
    with open(f"{save_dir}/ds_config.json", "w") as f:
        print(json.dumps(ds_config, indent=4), file=f)
    return ds_config, save_dir


def get_args():
    """获得参数"""
    parser = argparse.ArgumentParser()
    # train params
    parser.add_argument("--max_epochs", type=int, help="max epoches to run dataloader")
    parser.add_argument("--max_steps", type=int, help="max steps to run dataloader")
    parser.add_argument("--save_steps", type=int, default=2000, help="how many steps to save a model")
    parser.add_argument("--model_path", type=str, help="the path to load model")
    parser.add_argument("--seed", type=int, default=123456, help="the random seed")
    # save params
    parser.add_argument("--output_dir", type=str, default="output", help="the dir to save outputs")
    parser.add_argument("--ckpt_dir", type=str, default="ckpt", help="the subdir to save ckpts")
    parser.add_argument("--save_name", type=str, default="", help="save model in: ./output_dir/save_name+time/ckpt_dir/save_name")
    # parser.add_argument("--corpus", type=str, default="vicuna-7b")
    # lora params
    parser.add_argument("--use_lora", action="store_true", help="Whether to use LoRa")
    parser.add_argument("--load_lora", action="store_true", help="whether load ckpts")
    parser.add_argument("--load_lora_path", type=str, default="", help="the floader to load lora ckpts")
    # finetune params
    parser.add_argument("--add_token", type=bool, default=False, help="need to add special token")
    # distribute params
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    if args.save_name == "": 
        args.save_name = args.model_path.split("/")[-1]
    return args


def main():
    args = get_args()
    ds_config, save_dir = initialize_config(args)
    device = setup_distributed_environment(args.local_rank)
    model, tokenizer = initialize_model(args, device)  # 加载模型
    train_dataloader = prepare_dataloader(args, ds_config)
    train_model(model,device, tokenizer, train_dataloader, ds_config, args, save_dir)


if __name__ == "__main__":
    main()
