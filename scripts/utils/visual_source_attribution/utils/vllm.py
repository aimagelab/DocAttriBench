import subprocess
import time
from datetime import datetime
from typing import Any, Dict, List, Tuple, Union, Optional
import random
import os

from openai import OpenAI

from visual_source_attribution.utils.model_templates import MODEL_TEMPLATES
from visual_source_attribution.utils.vllm_settings import VLLM_SETTINGS

import logging


def deploy_vllm_model_from_settings(
    model_name: str,
    auto_port: bool = False,
    manual_port: Optional[int] = None
) -> Tuple[bool, str, Union[None, OpenAI]]:
    served_model_name = "_".join(model_name.split("/")[-2:])

    if served_model_name not in VLLM_SETTINGS:
        raise ValueError(
            f"Model name {served_model_name} is not " "associated with any settings."
        )

    model_settings = VLLM_SETTINGS[served_model_name]
    model_settings["model_name"] = model_name
    model_settings["target_message"] = "Application startup complete"

    if auto_port:
        deployed_models = os.listdir()
        deployed_models = list(filter(lambda x: x.endswith(".log"), deployed_models))
        occupied_ports = []
        for log_name in deployed_models:
            try:
                op = int(log_name.split("_")[-1].replace(".log", ""))
                occupied_ports.append(op)
            except Exception:
                continue
        
        new_port = model_settings["port"]
        while new_port in occupied_ports:
            new_port += random.randint(1, 20)
        model_settings["port"] = new_port
    elif manual_port is not None:
        model_settings["port"] = manual_port

    return deploy_vllm_model(**model_settings)


def deploy_vllm_model(
    model_name: str,
    num_gpus: int,
    target_message: str = "Application startup complete",
    max_dep_time: int = 2400,
    port: int = 8000,
    gpu_memory_utilization: float = 0.95,
    max_model_length: int = 6144,
    max_num_seqs: int = 32,
    max_num_batched_tokens: int = 8196,
    host: str = "0.0.0.0",
    dtype: Union[str, None] = None,
) -> Tuple[bool, str]:
    served_model_name = "_".join(model_name.split("/")[-2:])

    cmd1 = """pkill -f "vllm.entrypoints.openai.api_server .*--port {port}" || true""".format(
        **{
            "port": port,
        }
    )

    if num_gpus == 1:
        cmd2 = """export TOKENIZERS_PARALLELISM=true

python -m vllm.entrypoints.openai.api_server \
    --model {model_name} \
    --served-model-name {served_model_name} \
    --host {host} \
    --port {port} \
    --gpu-memory-utilization {gpu_memory_utilization} \
    --max-model-len {max_model_length} \
    --max-num-seqs {max_num_seqs} \
    --max-num-batched-tokens {max_num_batched_tokens} \
    --disable-log-stats \
    --trust-remote-code > {served_model_name}_{port}.log 2>&1 &

sleep 3; tail -n 80 {served_model_name}_{port}.log""".format(
            **{
                "model_name": model_name,
                "served_model_name": served_model_name,
                "port": port,
                "gpu_memory_utilization": gpu_memory_utilization,
                "max_model_length": max_model_length,
                "max_num_seqs": max_num_seqs,
                "max_num_batched_tokens": max_num_batched_tokens,
                "host": host,
            }
        )
    else:
        cmd2 = """export TOKENIZERS_PARALLELISM=true

python -m vllm.entrypoints.openai.api_server \
    --served-model-name {served_model_name} \
    --model {model_name} \
    --host {host} \
    --port {port} \
    --tensor-parallel-size {num_gpus} \
    --gpu-memory-utilization {gpu_memory_utilization} \
    --max-model-len {max_model_length} \
    --max-num-seqs {max_num_seqs} \
    --max-num-batched-tokens {max_num_batched_tokens} \
    --disable-log-stats \
    --trust-remote-code > {served_model_name}_{port}.log 2>&1 &

sleep 3; tail -n 80 {served_model_name}_{port}.log""".format(
            **{
                "model_name": model_name,
                "served_model_name": served_model_name,
                "num_gpus": num_gpus,
                "port": port,
                "gpu_memory_utilization": gpu_memory_utilization,
                "max_model_length": max_model_length,
                "max_num_seqs": max_num_seqs,
                "max_num_batched_tokens": max_num_batched_tokens,
                "host": host,
            }
        )

    if dtype is not None:
        cmd2 = cmd2.replace("--model", f"--dtype {dtype} --model")

    t0 = datetime.now()

    subprocess.run(cmd1, shell=True)

    time.sleep(3)

    subprocess.run(cmd2, shell=True)

    while True:
        time.sleep(5)

        with open(f"{served_model_name}_{port}.log", "r") as log_file:
            log_content = log_file.read()

        # Check if the target message is in the log content
        if target_message in log_content:
            break

        t1 = datetime.now()

        diff = t1 - t0
        if diff.seconds >= max_dep_time:
            return False, served_model_name, None

    client = OpenAI(base_url=f"http://127.0.0.1:{port}/v1", api_key="EMPTY")

    return True, served_model_name, client, port


class PromptVllmModel:
    def __init__(
        self,
        model_name: str,
        client: OpenAI,
        max_tokens: int = 256,
        temperature: float = 0.0,
        task: str = "answerBox",
    ) -> None:
        self.model_name = model_name

        assert task in MODEL_TEMPLATES, f"Task {task} is not implemented."
        assert model_name in MODEL_TEMPLATES[task], f"Model {model_name} is not implemented."

        self.system_template = MODEL_TEMPLATES[task][model_name].get("system_template", None)
        self.user_template = MODEL_TEMPLATES[task][model_name]["user_template"]
        self.image_args = MODEL_TEMPLATES[task][model_name].get("image_args", [])

        self.client = client
        self.max_tokens = max_tokens
        self.temperature = temperature

        self.task = task

        assert task in ["answerBox", "box", "boxFromAnswer"]

    def prepare_messages(self, **kwargs) -> List[Dict[str, Any]]:
        # ---- system and user prompts ----

        if self.system_template is not None:
            system_prompt = self.system_template.format(**kwargs)
        else:
            system_prompt = None
        user_prompt = self.user_template.format(**kwargs)

        # ---- image message ----

        img_message = {
            "type": "image_url",
            "image_url": {"url": kwargs["url"]},
        }

        for key in self.image_args:
            img_message[key] = kwargs[key]

        # ---- complete message ----

        if self.system_template is not None:
            messages = [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        img_message,
                    ],
                },
            ]
        else:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        img_message,
                    ],
                },
            ]

        return messages

    def __call__(self, **kwargs) -> str:
        messages = self.prepare_messages(**kwargs)

        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        return resp.choices[0].message.content
