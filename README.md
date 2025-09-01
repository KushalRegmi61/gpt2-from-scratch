# Building the LLM from Scratch

## Overview

This repository provides a step-by-step implementation of the fundamental components that power Large Language Models (LLMs). The goal is to build each piece from scratch tokenization, embeddings, data pipelines, attention, and transformers  to gain a practical, bottom-up understanding of how modern LLMs work.

LLMs are the backbone of today’s generative AI systems, enabling applications like chatbots, translation, summarization, and more. By reconstructing their components in code, this project bridges theory and practice for learners and practitioners.

## Features Implemented

* **Tokenizer**: Custom tokenizer with support for special tokens (`<|endoftext|>`, `<unk>`) and token-to-ID mapping.
* **Byte Pair Encoding (BPE)**: Subword tokenization algorithm implemented from scratch.
* **Data Handling**: Creation of input–target pairs using PyTorch `Dataset` and `DataLoader`; includes context size, stride, and batching.
* **Embeddings**: Token embeddings and absolute positional embeddings, combined to form input representations.

## Repository Structure

```
LICENSE
README.md
data/
    the-verdict.txt        # Sample dataset for experiments
notebooks/
    llm-from-scratch.ipynb # Step-by-step implementations in Jupyter
```

## Roadmap

Planned future extensions:

* Relative positional embeddings
* Attention mechanism
* Transformer encoder/decoder blocks
* Training toy LLMs on small datasets
* Finetuning for downstream tasks

## Getting Started

### Prerequisites

* Python 3.9+
* [PyTorch](https://pytorch.org/)
* Jupyter Notebook
* Standard libraries: `numpy`, `tqdm`

### Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/<your-username>/building-llm-from-scratch.git
cd building-llm-from-scratch
pip install -r requirements.txt
```

### Usage

Launch the notebook to explore step-by-step implementations:

```bash
jupyter notebook notebooks/llm-from-scratch.ipynb
```

## Contributing

Contributions are welcome! If you’d like to add new components, improve implementations, or extend documentation, please open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).

---

