<h1 align="center">
  [BMVC 2026] DocAttriBench: Benchmarking Answer Grounding in Document Visual Question Answering
</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2609.20574">
    <img src="https://img.shields.io/badge/Paper-arxiv.2508.20181-B31B1B.svg" alt="Paper">
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

## Abstract
Answer grounding in document visual question answering remains an open challenge: most benchmarks lack grounding annotations or provide limited-quality labels, while constructing grounded datasets still requires costly manual effort. We introduce DocAttriBench (DAB), a large-scale benchmark for fine-grained, element-level source attribution in Document VQA, grounding answers to specific layout elements such as text blocks, tables, and images. To build DAB, we propose a Mask-based Perplexity-Derived Attribution method (MAPPET) that combines document layout and language modeling to identify the most informative element for each answer. MAPPET measures the increase in perplexity after masking candidate elements and attributes the answer to the element contributing most to model confidence. Applying MAPPET to multiple existing Document VQA datasets yields DAB, with 237k documents and 296k question-answer pairs with element-level grounding. We benchmark grounding-capable multimodal LLMs on DAB, evaluating answer accuracy, attribution accuracy, and overall answer quality. Results show that while larger models generally achieve higher answer accuracy, even the strongest models often fail to localize the supporting elements. DAB provides a scalable benchmark for developing grounded, verifiable, and trustworthy Document VQA models. Dataset and code are available at https://aimagelab.github.io/DocAttriBench/.
