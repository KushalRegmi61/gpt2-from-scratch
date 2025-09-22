# Building a GPT-2 124M LLM from Scratch

## Overview

This repository provides a **step-by-step implementation of a GPT-2 124M model from scratch in PyTorch**. The model configuration is:

* Vocabulary size: 50,257  
* Context length: 1,024 tokens  
* Embedding dimension: 768  
* Number of attention heads: 12  
* Number of transformer layers: 12  
* Dropout rate: 0.1  
* Query-Key-Value bias: False  

The project reconstructs all fundamental components—tokenization, embeddings, attention mechanisms, feedforward networks, residual connections, transformer blocks, and the training loop—to gain a **deep, practical understanding of GPT-style LLMs**.

GPT-2 124M is a fully functional, manageable autoregressive transformer model. By building it from scratch, this project demonstrates **how embeddings, attention, normalization, and residuals interact** to generate coherent text.  

In addition to building the model, I plan to:

* **Fine-tune the model for classification tasks**, e.g., sentiment analysis, topic classification, or grievance categorization.  
* **Perform instruction fine-tuning** using PEFT (Parameter-Efficient Fine-Tuning) techniques to adapt the model for instruction-following or task-specific behavior without full model retraining.  

This workflow bridges **foundational model engineering** with **practical downstream adaptation**, giving insight into both how LLMs are constructed and how they are specialized for real-world tasks.

---

## Features Implemented

* **Tokenizer**: Custom tokenizer with support for special tokens (`<|endoftext|>`, `<unk>`) and token-to-ID mapping.  
* **Byte Pair Encoding (BPE)**: Subword tokenization implemented from scratch.  
* **Data Handling**: Input–target pair creation using PyTorch `Dataset` and `DataLoader`, including context size, stride, and batching.  
* **Embeddings**: Token embeddings and absolute positional embeddings combined for input representation.  
* **Attention Mechanisms**:  
  - Scaled Dot-Product Attention (mathematical + PyTorch implementation)  
  - Self-Attention with trainable Q, K, V matrices  
  - Causal Attention for autoregressive modeling  
  - Multi-Head Attention with projection and reshaping logic  
* **FeedForward Network**: Expansion–compression MLP with GELU activation.  
* **Layer Normalization**: Implemented from scratch for stable training.  
* **Residual (Shortcut) Connections**: Added for gradient flow across deep transformer stacks.  
* **Transformer Blocks**: Complete blocks combining attention, feedforward, normalization, and residuals.  
* **GPT-2 Architecture**: Stacked 12 transformer layers with uniform embedding/output dimensions.  
* **Weight Tying**: Input embeddings reused in output projection layer to reduce parameters.  
* **Autoregressive Text Generation**: Softmax-based token selection with multiple decoding strategies (argmax, sampling, top-k, nucleus).  
* **Training Loop**: End-to-end training with cross-entropy loss, AdamW optimizer, evaluation, and live sample generation.  
* **Metrics & Insights**: Tracking perplexity, training vs validation loss to monitor overfitting and model generalization.  


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
git clone https://github.com/kushalregmi61/building-llm-from-scratch.git
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

