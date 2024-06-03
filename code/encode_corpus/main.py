import pickle

fp = open("1.txt", "w")

fn = "/data1/dcy/refine-llama2-7b-20240315/code_example/train/data/llama2_7b_20230315/llama2_qwen14b_output_data_file.pkl"
wizard_fn = "/data1/dcy/refine-llama2-7b-20240315/yyz/encode_corpus/output/wizard/wizard_0.pkl"
# with open(fn, "rb") as f:
#     data = pickle.load(f)


with open(wizard_fn, "rb") as f:
    for i in range(3):
        data = pickle.load(f)
        # print(data, file=fp)
        for x in data[0]:
            print(x, end=" ", file=fp)
        print("", file=fp)
        for x in data[1]:
            print(x, end=" ", file=fp)
        print("", file=fp)  
