import json, os
from matplotlib import pyplot as plt
import numpy as np

# 运行前只需要改这个ckpt路径即可
ckpt_dir = "/data1/dcy/projects/fine-tune/fine-tune-yyz/train/output/5-1_00:12_Meta-Llama-3-8B-Instruct/ckpt/Meta-Llama-3-8B-Instruct_10000"

if __name__ == "__main__":
    # 读取loss_list
    loss_fn = os.path.join(ckpt_dir, "loss_list.json")
    with open(loss_fn, "r") as f:
        l = json.load(f)

    # 折线图保留n_points个点，否则上万条数据的折线图会很乱
    x, y = [], []
    n_points = 100  # 图上保留几个点
    step = len(l) // n_points 
    for i in range(0, len(l), step):
        x.append(i + 1)
        y.append(np.mean(l[i:i + step]))  # 取一段loss的平均值，而非单点取值

    # 绘制loss图片
    plt.title("train loss")
    plt.xlabel("steps")
    plt.ylabel("loss")
    plt.plot(x, y)

    # 将loss图片保存在对应checkpoint文件夹中
    fn = os.path.join(ckpt_dir, ckpt_dir.split("/")[-1] + "_loss.png")
    print("save loss picture at:", fn)
    plt.savefig(fn)