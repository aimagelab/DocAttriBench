from datetime import datetime
from typing import Tuple

import subprocess
import random
import time


def deploy_vllm_model(
    model_name, num_gpus, max_dep_time, target_message
) -> Tuple[bool, str]:
    served_model_name = "_".join(model_name.split("/")[-2:])

    log_id = random.randint(1000, 9999)
    model_log_path = f"{served_model_name}_{log_id}.log"
    while os.path.exists(model_log_path):
        time.sleep(3)
        log_id = random.randint(1000, 9999)
        model_log_path = f"{served_model_name}_{log_id}.log"
    

    if num_gpus == 1:
        cmd = """export TOKENIZERS_PARALLELISM=true

python -m vllm.entrypoints.openai.api_server \
    --model {model_name} \
    --served-model-name {served_model_name} \
    --dtype bfloat16 \
    --host 0.0.0.0 --port 8000 \
    --gpu-memory-utilization 0.95 \
    --max-model-len 6144 \
    --max-num-seqs 32 \
    --max-num-batched-tokens 8196 \
    --disable-log-stats \
    --trust-remote-code > {model_log_path} 2>&1 &

sleep 3; tail -n 80 {model_log_path}""".format(
            **{
                "model_name": model_name,
                "served_model_name": served_model_name,
                "model_log_path": model_log_path,
            }
        )
    else:
        cmd = """export TOKENIZERS_PARALLELISM=true

python -m vllm.entrypoints.openai.api_server \
    --served-model-name {served_model_name} \
    --model {model_name} \
    --dtype bfloat16 \
    --host 0.0.0.0 --port 8000 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 6144 \
    --max-num-seqs 8 \
    --max-num-batched-tokens 2048 \
    --tensor-parallel-size {num_gpus} \
    --disable-log-stats \
    --trust-remote-code > {model_log_path} 2>&1 &

sleep 3; tail -n 80 {model_log_path}""".format(
            **{
                "model_name": model_name,
                "served_model_name": served_model_name,
                "model_log_path": model_log_path,
                "num_gpus": num_gpus,
            }
        )

    t0 = datetime.now()

    subprocess.run(cmd, shell=True, check=True)

    while True:
        time.sleep(5)

        with open(f"{served_model_name}.log", "r") as log_file:
            log_content = log_file.read()

        # Check if the target message is in the log content
        if target_message in log_content:
            break

        t1 = datetime.now()

        diff = t1 - t0
        if diff.seconds >= max_dep_time:
            return False, served_model_name

    return True, served_model_name
