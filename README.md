<h1 align="center">
  [BMVC 2026] DocAttriBench: Benchmarking Answer Grounding in Document Visual Question Answering
</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2609.20574">
    <img src="https://img.shields.io/badge/Paper-arxiv.2609.20574-B31B1B.svg" alt="Paper">
  </a>
  <a href="https://aimagelab.github.io/DocAttriBench/">
    <img src="https://img.shields.io/badge/🌐-Project%20Page-blue.svg" alt="Project Page">
  </a>
  <a href="https://huggingface.co/collections/aimagelab/docattribench">
    <img src="https://img.shields.io/badge/🤗-HF%20Collection-yellow.svg" alt="HF Collection">
  </a>
</p>

This repository contains the reference code for the paper [DocAttriBench: Benchmarking Answer Grounding in Document Visual Question Answering](https://arxiv.org/abs/2609.20574), **BMVC 2026**.

## 📢 Latest Updates
  - **[2026/08/17]** Repo work in progress!
  - **[2026/08/22]** Dataset available in the Hugging Face collection!

## Citation

If you use this code, please cite our BMVC 2026 paper:

```bibtex

```

## Overview
Answer grounding in document visual question answering remains an open challenge: most benchmarks lack grounding annotations or provide limited-quality labels, while constructing grounded datasets still requires costly manual effort. We introduce DocAttriBench (DAB), a large-scale benchmark for fine-grained, element-level source attribution in Document VQA, grounding answers to specific layout elements such as text blocks, tables, and images. To build DAB, we propose a Mask-based Perplexity-Derived Attribution method (MAPPET) that combines document layout and language modeling to identify the most informative element for each answer. MAPPET measures the increase in perplexity after masking candidate elements and attributes the answer to the element contributing most to model confidence. Applying MAPPET to multiple existing Document VQA datasets yields DAB, with 237k documents and 296k question-answer pairs with element-level grounding. We benchmark grounding-capable multimodal LLMs on DAB, evaluating answer accuracy, attribution accuracy, and overall answer quality. Results show that while larger models generally achieve higher answer accuracy, even the strongest models often fail to localize the supporting elements. DAB provides a scalable benchmark for developing grounded, verifiable, and trustworthy Document VQA models. Dataset and code are available at https://aimagelab.github.io/DocAttriBench/.

## Table of Contents

- [1. Environment Setup](#1-environment-setup)
- [2. Conda Environment Installation](#2-conda-environment-installation)
- [3. Pipeline](#3-pipeline)
  - [3.1 Dataset Download](#31-dataset-download)
  - [3.2 Project Models Download](#32-project-models-download)
  - [3.3 Additional Models for Benchmarking and Processing](#33-additional-models-for-benchmarking-and-processing)
    - [Benchmark Inference and Optional Fine-Tuning](#benchmark-inference-and-optional-fine-tuning)
    - [Ground-Truth Generation](#ground-truth-generation)
    - [Answer-Concept Processing](#answer-concept-processing)
  - [3.4 Model Fine-Tuning](#34-model-fine-tuning)
  - [3.5 Model Merging](#35-model-merging)
  - [3.6 Inference](#36-inference)
    - [Reproducing Benchmark Scores with Additional Models](#reproducing-benchmark-scores-with-additional-models)
  - [3.7 Ground-Truth Generation](#37-ground-truth-generation)
  - [3.8 Answer Concepts Generation](#38-answer-concepts-generation)
  - [3.9 Final Metrics](#39-final-metrics)

# Enviroment Setup and Execution Pipeline

## 1. Environment Setup

The code of this repository was tested using these software versions:

| Component | Version |
| --- | --- |
| Python | **3.10.19** |
| CUDA runtime packages | **12.8** (`nvidia-cuda-runtime-cu12==12.8.90`) | 
| vLLM | **0.11.0** |
| PyTorch | **2.8.0** |
| Transformers / PEFT | **4.57.0 / 0.17.1** |
| FlashAttention | **2.8.3** |

All runs were launched in a test environment that complies with the software versions listed in the table. To ensure reproducibility, you can install the **`dab`** environments by following the instructions in the section below.

## 2. Conda Environment Installation

In order to properly replicate the production environment used by the code, you can run a dedicated sh script to create the **`dab`** environment. The steps are as follows: 

```bash
cd DocAttriBench/
bash install_env/create_dab_env.sh
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dab
```

The installer creates the environment named **`dab`** using `install_env/conda-explicit-spec.txt`, then installs the pinned pip dependencies with `--no-deps`. It verifies Python 3.10.19 and a total of 311 Conda-listed packages. `install_env/conda-list.txt` is a reference inventory, not an installation input. There is no environment YAML to install.

The installer refuses to modify an existing `dab` environment. If it already exists, activate it and inspect its versions. If installation fails midway, the script is not an automatic repair command for the partially created environment.

For reference, these are the installation operations inside the wrapper; use them as an alternative, **not after running the installer**:

```bash
conda create --yes --name dab --file install_env/conda-explicit-spec.txt
conda run --name dab python -m pip install \
  --disable-pip-version-check --no-deps \
  --requirement install_env/requirements-pip.txt
conda activate dab
```

No additional project dependency installation step is specified. Check the installed environment on your GPU machine with:

```bash
python --version
python -m pip show vllm torch transformers peft nvidia-cuda-runtime-cu12
python -c 'import torch; print("PyTorch CUDA build:", torch.version.cuda); print("CUDA available:", torch.cuda.is_available())'
```

## 3. Pipeline

The following pipeline covers the complete workflow for retrieving the DocAttriBench dataset and MAPPET models, optionally fine-tuning the models on DocAttriBench as described in the paper, running inference, and evaluating the results using the proposed metrics.

Dependent stages should be executed sequentially, with each job completed before its outputs are used by subsequent stages. Fine-tuning and model merging can be skipped when evaluating an already available compatible model, while reference-answer extraction can be performed independently of model inference.

The pipeline consists of the following steps:

1. Download the DocAttriBench dataset.
2. Download the required models.
3. *(Optional)* Fine-tune the model and merge the resulting weights.
4. Process concepts extracted from the ground-truth answers.
5. Run inference and process concepts extracted from the predicted answers.
6. Compute the evaluation metrics.

### 3.1 Dataset Download

The **DocAttriBench** dataset is publicly available on Hugging Face:

- **Dataset:** [aimagelab/DocAttriBench](https://huggingface.co/datasets/aimagelab/DocAttriBench)
- **Collection:** [aimagelab/docattribench](https://huggingface.co/collections/aimagelab/docattribench)

Download the dataset into the `/data` directory of this repository. From the repository root, run:

```bash
huggingface-cli download aimagelab/DocAttriBench \
    --repo-type dataset \
    --local-dir data/DocAttriBench
```

Alternatively, the dataset can be downloaded manually from its Hugging Face page and placed under:

```text
data/DocAttriBench/
```

All subsequent steps assume that the dataset is available at this location.

### 3.2 Project Models Download

The three **MAPPET** models released for DocAttriBench are available through the official Hugging Face collection:

[https://huggingface.co/collections/aimagelab/docattribench](https://huggingface.co/collections/aimagelab/docattribench)

The individual model repositories are:

- [aimagelab/Mappet-7B](https://huggingface.co/aimagelab/Mappet-7B)
- [aimagelab/Mappet-8B](https://huggingface.co/aimagelab/Mappet-8B)
- [aimagelab/Mappet-I-8B](https://huggingface.co/aimagelab/Mappet-I-8B)

Download the required models into the `/models` directory of this repository.

For example, from the repository root:

```bash
huggingface-cli download aimagelab/Mappet-7B \
    --local-dir models/Mappet-7B

huggingface-cli download aimagelab/Mappet-8B \
    --local-dir models/Mappet-8B

huggingface-cli download aimagelab/Mappet-I-8B \
    --local-dir models/Mappet-I-8B
```

After downloading, the expected directory structure is:

```text
models/
├── Mappet-7B/
├── Mappet-8B/
└── Mappet-I-8B/
```

All subsequent inference and evaluation steps assume that the required MAPPET models are available under the repository's `/models` directory.

### 3.3 Additional Models for Benchmarking and Processing

In addition to the released MAPPET models described in the previous section, the pipeline relies on a set of external vision-language models for benchmark reproduction, optional fine-tuning, ground-truth generation, and answer-concept processing.

Unless otherwise specified, these models should be downloaded from Hugging Face and stored under the repository's `/models` directory.

#### Benchmark Inference and Optional Fine-Tuning

The following models are used to reproduce the benchmark results reported in the paper. They can also be used as starting checkpoints for the optional fine-tuning stage.

| Model | Hugging Face |
|---|---|
| `Qwen2.5-VL-3B-Instruct` | [Qwen/Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) |
| `Qwen2.5-VL-7B-Instruct` | [Qwen/Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) |
| `Qwen2.5-VL-32B-Instruct` | [Qwen/Qwen2.5-VL-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-32B-Instruct) |
| `Qwen3-VL-2B-Instruct` | [Qwen/Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) |
| `Qwen3-VL-8B-Instruct` | [Qwen/Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) |
| `Qwen3-VL-32B-Instruct` | [Qwen/Qwen3-VL-32B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct) |
| `InternVL2.5-2B` | [OpenGVLab/InternVL2_5-2B](https://huggingface.co/OpenGVLab/InternVL2_5-2B) |
| `InternVL2.5-8B` | [OpenGVLab/InternVL2_5-8B](https://huggingface.co/OpenGVLab/InternVL2_5-8B) |
| `InternVL2.5-38B` | [OpenGVLab/InternVL2_5-38B](https://huggingface.co/OpenGVLab/InternVL2_5-38B) |
| `InternVL3-2B-Instruct` | [OpenGVLab/InternVL3-2B-Instruct](https://huggingface.co/OpenGVLab/InternVL3-2B-Instruct) |
| `InternVL3-8B-Instruct` | [OpenGVLab/InternVL3-8B-Instruct](https://huggingface.co/OpenGVLab/InternVL3-8B-Instruct) |
| `InternVL3-38B-Instruct` | [OpenGVLab/InternVL3-38B-Instruct](https://huggingface.co/OpenGVLab/InternVL3-38B-Instruct) |
| `visa-7B-single-fulldata` | [MrLight/visa-7B-single-fulldata](https://huggingface.co/MrLight/visa-7B-single-fulldata) |

> **Note:** `MrLight/visa-7B-single-fulldata` is distributed as a PEFT/LoRA adapter and requires the corresponding base model, [`Qwen/Qwen2-VL-7B-Instruct`](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct), to be downloaded before merging. When VISA is included in the benchmark, make sure the base model is available under `/models`, then use the dedicated VISA model-merging job described in the model-merging section.

#### Ground-Truth Generation

Ground-truth answer-concept extraction uses the AWQ-quantized 72B Qwen2.5-VL model:

| Purpose | Model | Hugging Face |
|---|---|---|
| Ground-truth answer-concept generation | `Qwen2.5-VL-72B-Instruct-AWQ` | [Qwen/Qwen2.5-VL-72B-Instruct-AWQ](https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct-AWQ) |

This model is used by the ground-truth processing stage described in **Section 3.7**.

#### Answer-Concept Processing

Predicted answers generated during inference are processed with the AWQ-quantized 32B Qwen2.5-VL model:

| Purpose | Model | Hugging Face |
|---|---|---|
| Predicted answer-concept extraction | `Qwen2.5-VL-32B-Instruct-AWQ` | [Qwen/Qwen2.5-VL-32B-Instruct-AWQ](https://huggingface.co/Qwen/Qwen2.5-VL-32B-Instruct-AWQ) |

This model is used by the answer-concept processing stage described in **Section 3.8**.

Models can be downloaded with the Hugging Face CLI. For example:

```bash
huggingface-cli download Qwen/Qwen2.5-VL-3B-Instruct \
    --local-dir models/Qwen2.5-VL-3B-Instruct
```

Use the corresponding Hugging Face repository identifier and target directory for each additional model required by the workflow.

### 3.4 Model Fine-Tuning

The models can be fine-tuned on **DocAttriBench** using **LoRA**. The fine-tuning implementation is provided in `scripts/finetuning/`, while the corresponding SLURM job files are available under `jobs/finetuning/`.

Two fine-tuning scripts are used depending on the model family. Both Qwen models share the same implementation, whereas InternVL uses a dedicated script:

| Model | Fine-tuning script | SLURM job |
|---|---|---|
| `Qwen/Qwen2.5-VL-7B-Instruct` | `scripts/finetuning/finetuning_qwen.py` | `jobs/finetuning/qwen25_finetuning.sh` |
| `Qwen/Qwen3-VL-8B-Instruct` | `scripts/finetuning/finetuning_qwen.py` | `jobs/finetuning/qwen3_finetuning.sh` |
| `OpenGVLab/InternVL3_5-8B-Instruct` | `scripts/finetuning/finetuning_internvl.py` | `jobs/finetuning/internvl35_finetuning.sh` |

Each SLURM script is configured to launch the corresponding fine-tuning procedure with the appropriate model-specific settings.

From the repository root, a fine-tuning job can be submitted with:

```bash
sbatch jobs/finetuning/qwen25_finetuning.sh
```

Similarly, use:

```bash
sbatch jobs/finetuning/qwen3_finetuning.sh
```
for `Qwen/Qwen3-VL-8B-Instruct`, or:

```bash
sbatch jobs/finetuning/internvl35_finetuning.sh
```

for `OpenGVLab/InternVL3_5-8B-Instruct`.

The fine-tuning stage is optional and can be skipped when evaluating an already available compatible model or when using previously generated LoRA adapters. The resulting adapters can then be used in the subsequent model-merging step before running inference.

### 3.5 Model Merging

After fine-tuning, the resulting LoRA adapters can be merged with their corresponding base models to obtain standalone models that can be used directly during inference.

The model-merging logic is implemented in:

```text
scripts/model_merge/model_merge.py
```

The corresponding SLURM job scripts are located in:

```text
jobs/model_merge/
```

The available merging jobs are:

| Model | SLURM job |
|---|---|
| MAPPET based on Qwen3-VL | `jobs/model_merge/mappet3_model_merge.sh` |
| MAPPET based on Qwen2.5-VL | `jobs/model_merge/mappet25_model_merge.sh` |
| MAPPET based on InternVL | `jobs/model_merge/mappetintern_model_merge.sh` |
| VISA | `jobs/model_merge/VISA_model_merge.sh` |

Each SLURM job invokes `scripts/model_merge/model_merge.py` with the appropriate base model, LoRA adapter, and output configuration.

For example, to merge the Qwen2.5-based MAPPET model, run from the repository root:

```bash
sbatch jobs/model_merge/mappet25_model_merge.sh
```

Similarly, use the corresponding SLURM script for the Qwen3-VL- or InternVL-based MAPPET variants.

A dedicated SLURM job is also provided for the **VISA** model:

```bash
sbatch jobs/model_merge/VISA_model_merge.sh
```

This job is included to support the use of VISA as an additional baseline for benchmarking within the same evaluation pipeline.

The model-merging stage is optional and can be skipped when the model to be evaluated is already available in a fully merged and compatible format.

### 3.6 Inference

Model inference is handled by:

```text
scripts/inference/answer_inference.py
```

This script supports the three inference tasks used in the evaluation pipeline:

| Task | Description |
|---|---|
| `answerBox` | Generate an answer together with the supporting bounding boxes. |
| `box` | Localize the visual evidence using only the question and the input image. |
| `boxFromAnswer` | Localize the visual evidence using the question, the input image, and the dataset reference answer. |

The corresponding SLURM job arrays are available under:

```text
jobs/inference/
```

Each job array runs the selected inference task across all three released MAPPET models:

- `Mappet-7B`
- `Mappet-8B`
- `Mappet-I-8B`

The available wrappers are:

| Wrapper | Task | Requested behavior |
|---|---|---|
| `jobs/inference/answerBox_inference.sh` | `answerBox` (default) | Generate an answer with supporting boxes |
| `jobs/inference/box_inference.sh` | `box` | Locate evidence from the question and image |
| `jobs/inference/boxFromAnswer_inference.sh` | `boxFromAnswer` | Locate evidence given the question, image, and dataset reference answer |

To launch inference, submit the wrapper corresponding to the desired task from the repository root.

For example, to generate answers together with supporting boxes:

```bash
sbatch jobs/inference/answerBox_inference.sh
```

To run evidence localization from the question and image only:

```bash
sbatch jobs/inference/box_inference.sh
```

To localize evidence using the dataset reference answer in addition to the question and image:

```bash
sbatch jobs/inference/boxFromAnswer_inference.sh
```

Because these wrappers are implemented as SLURM job arrays, a single submission covers all three MAPPET models using the task-specific configuration defined in the corresponding job file.

#### Reproducing Benchmark Scores with Additional Models

For reproducibility purposes, additional SLURM job arrays are provided to run the same three inference tasks on the baseline models used for benchmark score reproduction.

These wrappers are available under:

```text
jobs/inference/other_models/
```

The available jobs are:

| Wrapper | Task |
|---|---|
| `jobs/inference/other_models/answerBox_inference.sh` | `answerBox` |
| `jobs/inference/other_models/box_inference.sh` | `box` |
| `jobs/inference/other_models/boxFromAnswer_inference.sh` | `boxFromAnswer` |

Each wrapper launches `scripts/inference/answer_inference.py` as a SLURM job array over the following models:

- `MrLight/visa-7B-single-fulldata-merged`
- `OpenGVLab/InternVL2_5-2B`
- `OpenGVLab/InternVL2_5-8B`
- `OpenGVLab/InternVL2_5-38B`
- `OpenGVLab/InternVL3-2B-Instruct`
- `OpenGVLab/InternVL3-8B-Instruct`
- `OpenGVLab/InternVL3-38B-Instruct`
- `Qwen/Qwen2.5-VL-3B-Instruct`
- `Qwen/Qwen2.5-VL-7B-Instruct`
- `Qwen/Qwen2.5-VL-32B-Instruct`
- `Qwen/Qwen3-VL-2B-Instruct`
- `Qwen/Qwen3-VL-8B-Instruct`
- `Qwen/Qwen3-VL-32B-Instruct`

For example, to reproduce `answerBox` predictions across all benchmark models, run:

```bash
sbatch jobs/inference/other_models/answerBox_inference.sh
```

Similarly, use:

```bash
sbatch jobs/inference/other_models/box_inference.sh
```

for `box` inference, or:

```bash
sbatch jobs/inference/other_models/boxFromAnswer_inference.sh
```

for `boxFromAnswer` inference.

The VISA entry refers to the already merged model, `MrLight/visa-7B-single-fulldata-merged`, which should be prepared beforehand using the dedicated model-merging step.

The generated inference outputs are used by the subsequent answer-processing and evaluation stages of the pipeline.

### 3.7 Ground-Truth Generation

Ground-truth answer processing is handled by:

```text
scripts/ground_truth/gt_answer_processing.py
```

The corresponding SLURM job is:

```text
jobs/ground_truth/gt_answer_processing.sh
```

This stage extracts typed answer concepts from the **existing reference answers** contained in the dataset. It does **not** generate bounding-box ground truth and does not rerun the masking/perplexity-based attribution procedure described in the paper. Any grounding annotations required for evaluation must therefore already be available in the input dataset.

The script reads the test samples from:

```text
data/<dataset>/test/json/*.json
```

For each sample, it uses the `query` field together with the list-valued `answer` field to prompt a text-only concept extraction step. The resulting outputs are written to:

```text
data/<dataset>/test/vllm_extractive_answers.json
```

The output file contains a list of raw model-generated strings describing the extracted answer values and their associated types.

If the output file already exists, the corresponding processing step is skipped.

To run the ground-truth processing stage from the repository root, submit:

```bash
sbatch jobs/ground_truth/gt_answer_processing.sh
```

This stage is independent of model inference and can be executed before, after, or separately from the inference pipeline.




### 3.8 Answer Concepts Generation

This stage processes the model outputs produced by the `answerBox` inference task and extracts the answer concepts required by the evaluation pipeline.

Before running this step, generate the `answerBox` predictions as described in **Section 3.6**.

The processing logic is implemented in:

```text
scripts/processing/answer_processing.py
```

The corresponding SLURM job is:

```text
jobs/processing/answer_processing.sh
```

The script takes the inference outputs generated by the `answerBox` task as input and performs a concept-extraction step over the predicted answers.

For each inference file, the raw extraction strings are saved alongside the original prediction file. The output filename is obtained by replacing the `.json` extension with:

```text
_vllmExtraction.json
```

For example:

```text
predictions.json
```

produces:

```text
predictions_vllmExtraction.json
```

To run this processing stage from the repository root, submit:

```bash
sbatch jobs/processing/answer_processing.sh
```

The generated concept-extraction files are used by the subsequent evaluation steps to compare predicted answer concepts against the processed ground-truth concepts.


### 3.9 Final Metrics

> **Work in progress:** scripts for computing the final evaluation metrics will be added in a future version.
