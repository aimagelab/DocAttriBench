VLLM_SETTINGS = dict()


VLLM_SETTINGS["Qwen_Qwen2.5-VL-7B-Instruct-AWQ"] = {
    "host": "0.0.0.0",
    "port": 8001,
    "gpu_memory_utilization": 0.80,
    "max_model_length": 6144,
    "max_num_seqs": 4,
    "max_num_batched_tokens": 1024,
    "num_gpus": 1,
}


VLLM_SETTINGS["Qwen_Qwen2.5-VL-32B-Instruct-AWQ"] = {
    "host": "0.0.0.0",
    "port": 8002,
    "gpu_memory_utilization": 0.80,
    "max_model_length": 6144,
    "max_num_seqs": 4,
    "max_num_batched_tokens": 1024,
    "num_gpus": 2,
}


VLLM_SETTINGS["Qwen_Qwen2.5-VL-3B-Instruct"] = {
    "host": "0.0.0.0",
    "port": 8003,
    "gpu_memory_utilization": 0.80,
    "max_model_length": 6144,
    "max_num_seqs": 4,
    "max_num_batched_tokens": 1024,
    "num_gpus": 1,
    "dtype": "bfloat16",
}


VLLM_SETTINGS["Qwen_Qwen2.5-VL-7B-Instruct"] = {
    "host": "0.0.0.0",
    "port": 8004,
    "gpu_memory_utilization": 0.80,
    "max_model_length": 6144,
    "max_num_seqs": 4,
    "max_num_batched_tokens": 1024,
    "num_gpus": 1,
    "dtype": "bfloat16",
}


VLLM_SETTINGS["Qwen_Qwen3-VL-2B-Instruct"] = {
    "host": "0.0.0.0",
    "port": 8005,
    "gpu_memory_utilization": 0.80,
    "max_model_length": 6144,
    "max_num_seqs": 4,
    "max_num_batched_tokens": 1024,
    "num_gpus": 1,
    "dtype": "bfloat16",
}


VLLM_SETTINGS["Qwen_Qwen3-VL-8B-Instruct"] = {
    "host": "0.0.0.0",
    "port": 8007,
    "gpu_memory_utilization": 0.80,
    "max_model_length": 6144,
    "max_num_seqs": 4,
    "max_num_batched_tokens": 1024,
    "num_gpus": 1,
    "dtype": "bfloat16",
}


VLLM_SETTINGS["Qwen_Qwen3-VL-32B-Instruct-FP8"] = {
    "host": "0.0.0.0",
    "port": 8008,
    "gpu_memory_utilization": 0.80,
    "max_model_length": 6144,
    "max_num_seqs": 4,
    "max_num_batched_tokens": 1024,
    "num_gpus": 2,
    "dtype": "bfloat16",
}


VLLM_SETTINGS["Qwen_Qwen3-VL-32B-Instruct"] = {
    "host": "0.0.0.0",
    "port": 8008,
    "gpu_memory_utilization": 0.80,
    "max_model_length": 6144,
    "max_num_seqs": 4,
    "max_num_batched_tokens": 1024,
    "num_gpus": 4,
    "dtype": "bfloat16",
}


VLLM_SETTINGS["OpenGVLab_InternVL2_5-2B"] = {
    "host": "0.0.0.0",
    "port": 8009,
    "gpu_memory_utilization": 0.80,
    "max_model_length": 6144,
    "max_num_seqs": 4,
    "max_num_batched_tokens": 1024,
    "num_gpus": 1,
    "dtype": "bfloat16",
}


VLLM_SETTINGS["OpenGVLab_InternVL2_5-8B"] = {
    "host": "0.0.0.0",
    "port": 8010,
    "gpu_memory_utilization": 0.80,
    "max_model_length": 6144,
    "max_num_seqs": 4,
    "max_num_batched_tokens": 1024,
    "num_gpus": 1,
    "dtype": "bfloat16",
}


VLLM_SETTINGS["OpenGVLab_InternVL2_5-38B"] = {
    "host": "0.0.0.0",
    "port": 8011,
    "gpu_memory_utilization": 0.80,
    "max_model_length": 6144,
    "max_num_seqs": 4,
    "max_num_batched_tokens": 1024,
    "num_gpus": 1,
    "dtype": "bfloat16",
}


VLLM_SETTINGS["OpenGVLab_InternVL3-2B-Instruct"] = {
    "host": "0.0.0.0",
    "port": 8012,
    "gpu_memory_utilization": 0.80,
    "max_model_length": 6144,
    "max_num_seqs": 4,
    "max_num_batched_tokens": 1024,
    "num_gpus": 1,
    "dtype": "bfloat16",
}


VLLM_SETTINGS["OpenGVLab_InternVL3-8B-Instruct"] = {
    "host": "0.0.0.0",
    "port": 8013,
    "gpu_memory_utilization": 0.80,
    "max_model_length": 6144,
    "max_num_seqs": 4,
    "max_num_batched_tokens": 1024,
    "num_gpus": 1,
    "dtype": "bfloat16",
}


VLLM_SETTINGS["OpenGVLab_InternVL3-38B-Instruct"] = {
    "host": "0.0.0.0",
    "port": 8014,
    "gpu_memory_utilization": 0.80,
    "max_model_length": 6144,
    "max_num_seqs": 4,
    "max_num_batched_tokens": 1024,
    "num_gpus": 1,
    "dtype": "bfloat16",
}


VLLM_SETTINGS["OpenGVLab_InternVL3_5-2B-Instruct"] = {
    "host": "0.0.0.0",
    "port": 8015,
    "gpu_memory_utilization": 0.80,
    "max_model_length": 6144,
    "max_num_seqs": 4,
    "max_num_batched_tokens": 1024,
    "num_gpus": 1,
    "dtype": "bfloat16",
}


VLLM_SETTINGS["OpenGVLab_InternVL3_5-8B-Instruct"] = {
    "host": "0.0.0.0",
    "port": 8016,
    "gpu_memory_utilization": 0.80,
    "max_model_length": 6144,
    "max_num_seqs": 4,
    "max_num_batched_tokens": 1024,
    "num_gpus": 1,
    "dtype": "bfloat16",
}


VLLM_SETTINGS["OpenGVLab_InternVL3_5-38B-Instruct"] = {
    "host": "0.0.0.0",
    "port": 8017,
    "gpu_memory_utilization": 0.80,
    "max_model_length": 6144,
    "max_num_seqs": 4,
    "max_num_batched_tokens": 1024,
    "num_gpus": 1,
    "dtype": "bfloat16",
}
