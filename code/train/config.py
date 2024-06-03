from peft import LoraConfig, PromptTuningConfig, TaskType, PromptTuningInit

DS_CONFIG_ft = {  # finetune 使用的config
    "bf16": {
        "enabled": True, #
    },
    # "fp16": {
    #     "enabled": False,  # 确保禁用16位浮点（FP16）
    # },
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 5e-5,
            "betas": [0.98, 0.999],
            "eps": 1e-9
        }
    },
    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": 1e-5,
            "warmup_max_lr": 3e-5,
            "warmup_num_steps": 300
        }
    },
    "zero_optimization": { # https://www.bilibili.com/video/BV1fb421t7KN/?p=3&spm_id_from=pageDriver DeepSpeed优化器并行ZeRO1/2/3原理 #大模型 #分布式并行 #训练
        "stage": 1, # ZeRO优化级别 (1, 2, 3)，级别越高，节省的显存越多，但可能需要更多的计算资源或可能会有更高的通信开销。
        "allgather_partitions": True,  # 在更新参数之前，是否聚合（allgather）优化器状态的分区。这对于确保所有GPU都有完整的更新前的参数状态很重要
        "allgather_bucket_size": 2e8,  # 控制执行allgather操作时使用的bucket的大小(单位:字节)。较小的bucket可以减少峰值显存使用，但可能会增加通信次数。建议中等大小200-500MB
        "overlap_comm": True,  # 是否允许通信（如梯度allreduce）与计算重叠。启用这一选项可以提高训练效率，但在某些情况下可能会增加显存使用
        "reduce_scatter": True,  # 在更新参数之前，是否使用reduce scatter来减少梯度。这样做可以减少梯度聚合的显存需求
        "reduce_bucket_size": 2e8,  #  控制执行reduce scatter操作时使用的bucket大小。与allgather_bucket_size类似，较小的值可以减少显存峰值，但可能会导致更多的通信。
        "contiguous_gradients": True,  # 是否在内存中连续存储梯度。这可以提高一些操作的效率，但可能会增加总体的显存使用
        "stage3_gather_16bit_weights_on_model_save": True,
        "offload_optimizer": {
            "device": "cpu"  # 控制是否将优化器状态卸载到CPU
        }
    },
    
    "gradient_accumulation_steps": 256,
    "train_micro_batch_size_per_gpu": 1,
    "wall_clock_breakdown": False,
    "steps_per_print": 100,


}


DS_CONFIG_lora = {  # 使用lora时使用的config
    "bf16": {
        "enabled": True,
    },
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 3e-5,
            "betas": [0.98, 0.999],
            "eps": 1e-9
        }
    },
    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": 3e-5,
            "warmup_max_lr": 1e-4,
            "warmup_num_steps": 300
        }
    },
    "zero_optimization": {
        "stage": 1, # ZeRO优化级别 (1, 2, 3)，级别越高，节省的显存越多，但可能需要更多的计算资源或可能会有更高的通信开销。
        "allgather_partitions": True,  # 在更新参数之前，是否聚合（allgather）优化器状态的分区。这对于确保所有GPU都有完整的更新前的参数状态很重要
        "allgather_bucket_size": 2e8,  # 控制执行allgather操作时使用的bucket的大小(单位:字节)。较小的bucket可以减少峰值显存使用，但可能会增加通信次数。建议中等大小200-500MB
        "overlap_comm": True,  # 是否允许通信（如梯度allreduce）与计算重叠。启用这一选项可以提高训练效率，但在某些情况下可能会增加显存使用
        "reduce_scatter": True,  # 在更新参数之前，是否使用reduce scatter来减少梯度。这样做可以减少梯度聚合的显存需求
        "reduce_bucket_size": 2e8,  #  控制执行reduce scatter操作时使用的bucket大小。与allgather_bucket_size类似，较小的值可以减少显存峰值，但可能会导致更多的通信。
        "contiguous_gradients": True,  # 是否在内存中连续存储梯度。这可以提高一些操作的效率，但可能会增加总体的显存使用
        "stage3_gather_16bit_weights_on_model_save": True,
        "offload_optimizer": {
            "device": "cpu"  # 控制是否将优化器状态卸载到CPU
        }
    },
    "gradient_accumulation_steps": 256,
    "train_micro_batch_size_per_gpu": 1,
    "steps_per_print": 100, 
    "wall_clock_breakdown": False
}


LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
TARGET_MODULES = [
    "q_proj",
    "v_proj",
    # "gate_proj",
    # "down_proj",
    # "up_proj"
]

lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
)

# lora_config = PromptTuningConfig(
#     task_type=TaskType.CAUSAL_LM,
#     prompt_tuning_init=PromptTuningInit.RANDOM,
#     num_virtual_tokens=20,
#     # prompt_tuning_init_text="Classify if the tweet is a complaint or not:",
#     # tokenizer_name_or_path=model_name_or_path,
# )

template_pool = {
    'wround_woinput':[
            # "问：{}\n答：{}\n",
        "Instruction:{}\Response:{}\n",
        "{}\n{}\n"
    ],

}
meta_prompt = ""