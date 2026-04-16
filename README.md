# BERT Finetuning on IMDB Dataset

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Transformers 4.30+](https://img.shields.io/badge/Transformers-4.30+-yellow.svg)](https://huggingface.co/transformers/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Binary sentiment classification model fine-tuned on the IMDB movie review dataset using BERT-base-uncased. This repository provides a complete pipeline for training, evaluation, and deployment to Hugging Face Hub.

> **Model Card**: [Jim1892/IMDB-BERT-Finetuned](https://huggingface.co/Jim1892/IMDB-BERT-Finetuned) on 🤗 Hub

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Setup](#setup)
- [Usage](#usage)
- [Reproducibility](#reproducibility)
- [Troubleshooting](#troubleshooting)
- [References](#references)

## Overview

This project fine-tunes Google's BERT-base-uncased model for binary sentiment classification on the IMDB movie review dataset. The resulting model achieves strong performance on sentiment analysis tasks and is automatically pushed to Hugging Face Hub for easy deployment and sharing.

### Key Statistics

| Metric | Value |
|--------|-------|
| **Base Model** | bert-base-uncased |
| **Dataset** | IMDB Movie Reviews |
| **Task** | Binary Classification (Positive/Negative) |
| **Training Samples** | 25,000 (subset: 1,000 in notebook) |
| **Max Sequence Length** | 256 tokens |
| **Training Epochs** | 1 |
| **Batch Size** | 8 |
| **Learning Rate** | 2e-5 |
| **Weight Decay** | 0.01 |
| **Framework** | Hugging Face Transformers + PyTorch |
| **Hardware** | GPU recommended (CUDA 11.8+) |

## Quick Start

### Load and Use the Fine-Tuned Model

```python
from transformers import pipeline

# Load the fine-tuned model from Hub
classifier = pipeline("text-classification", 
                     model="Jim1892/IMDB-BERT-Finetuned")

# Make predictions
result = classifier("This movie was absolutely amazing!")
print(result)  # Output: [{'label': 'POSITIVE', 'score': 0.99}]
```

### Run the Full Training Notebook

```bash
# Install dependencies
pip install -r requirements.txt

# Launch Jupyter and open BERT_Finetuning.ipynb
jupyter notebook
```

## Setup

### Prerequisites

- Python 3.8+
- GPU with 4GB+ VRAM (or CPU with ~16GB RAM)
- pip or conda package manager

### Installation

1. **Clone the repository**

```bash
git clone https://github.com/NayeemHossenJim/HuggingFace-BERT-Optimization.git
cd HuggingFace-BERT-Optimization
```

1. **Create a virtual environment** (recommended)

```bash
# Using venv
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # macOS/Linux
```

1. **Install dependencies**

```bash
pip install --upgrade pip
pip install datasets fsspec torch transformers[torch]
pip install jupyter ipywidgets
```

Or install from a pinned requirements file (when available):

```bash
pip install -r requirements.txt
```

### Verify Installation

```python
import torch
import transformers
print(f"PyTorch: {torch.__version__}")
print(f"Transformers: {transformers.__version__}")
print(f"GPU Available: {torch.cuda.is_available()}")
```

## Usage

### 1. Training Pipeline

The notebook provides a complete training pipeline:

```python
from transformers import BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset

# 1. Load IMDB dataset
dataset = load_dataset("imdb")

# 2. Tokenize and preprocess
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
def tokenize_fn(examples):
    return tokenizer(examples["text"], padding="max_length", 
                    truncation=True, max_length=256)

# 3. Load model
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=2)

# 4. Configure training
training_args = TrainingArguments(
    output_dir="./bert-finetuned-imdb",
    num_train_epochs=1,
    per_device_train_batch_size=8,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_dir="./logs",
    push_to_hub=True,
    hub_model_id="<your-username>/IMDB-BERT-Finetuned",
    hub_strategy="end",
)

# 5. Train
trainer = Trainer(model=model, args=training_args, 
                  train_dataset=train_dataset, 
                  eval_dataset=test_dataset)
trainer.train()

# 6. Push to Hub
trainer.push_to_hub(commit_message="Upload fine-tuned IMDB BERT")
```

### 2. Inference

Use the fine-tuned model for sentiment prediction:

```python
from transformers import pipeline

classifier = pipeline("text-classification", 
                     model="Jim1892/IMDB-BERT-Finetuned")

# Single prediction
text = "This movie was amazing and I loved the acting!"
result = classifier(text)
print(result)

# Batch predictions
texts = [
    "Terrible movie, waste of time.",
    "Outstanding performance by the lead actor!"
]
results = classifier(texts)
for text, result in zip(texts, results):
    print(f"Text: {text}")
    print(f"Prediction: {result}\n")
```

### 3. Fine-grained Control

Access model outputs for custom logic:

```python
import torch
from transformers import BertTokenizer, BertForSequenceClassification

tokenizer = BertTokenizer.from_pretrained("Jim1892/IMDB-BERT-Finetuned")
model = BertForSequenceClassification.from_pretrained("Jim1892/IMDB-BERT-Finetuned")

text = "This film exceeded my expectations!"
inputs = tokenizer(text, return_tensors="pt", max_length=256, 
                   padding="max_length", truncation=True)

with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    
print(f"Predicted Label: {predictions.item()}")  # 1 = POSITIVE, 0 = NEGATIVE
```

## Reproducibility

### Exact Hyperparameters

To reproduce training results with the same hyperparameters:

```python
training_args = TrainingArguments(
    output_dir="./bert-finetuned-imdb",
    num_train_epochs=1,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_steps=500,
    logging_steps=100,
    eval_strategy="no",  # Disable eval during training
    save_strategy="no",
    report_to="none",
    seed=42,  # Set seed for reproducibility
    dataloader_pin_memory=True,
    fp16=True,  # Mixed precision (GPU only)
)
```

### Setting Random Seeds

For full reproducibility across runs:

```python
import random
import numpy as np
import torch

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)
```

### Pushing to Hugging Face Hub

1. **Authenticate first**

```python
from huggingface_hub import notebook_login

notebook_login()  # Or use: huggingface-cli login
```

1. **Configure TrainingArguments with Hub settings**

```python
training_args = TrainingArguments(
    # ... other args ...
    push_to_hub=True,
    hub_model_id="<your-username>/<model-name>",
    hub_strategy="end",  # Push only at end of training
    hub_token="<your-token>",  # Optional: explicitly pass token
)
```

1. **Push after training**

```python
trainer.push_to_hub(commit_message="Initial training run")
```

### Model Card

After pushing to Hub, update the model card by editing the README.md on the model page:

- Training data: IMDB Movie Reviews (25K subset)
- Model architecture: BERT-base-uncased
- Intended use: Binary sentiment classification
- Limitations: Trained on English movie reviews; may not generalize to other domains
- Dataset link: [IMDB Dataset on Hub](https://huggingface.co/datasets/imdb)

## Troubleshooting

### Windows: mmap Lock Error on Model Push

**Error**: `OSError: [WinError 1224] The file cannot be accessed by the system`

**Root Cause**: Windows locks the `model.safetensors` file when it's loaded in memory, preventing writes.

**Solution**:

- Use Trainer-managed Hub configuration (as shown in Usage section)
- **Do NOT** manually load the model from the output directory before pushing
- Always push via `trainer.push_to_hub()` after training completes
- If you need to clean up, delete the output directory only after successful Hub push

```python
# ✅ CORRECT - Use trainer push directly
trainer.push_to_hub(commit_message="Final model")

# ❌ AVOID - This can cause mmap locks on Windows
model = BertForSequenceClassification.from_pretrained("./bert-finetuned-imdb")
model.push_to_hub("my-model")
```

### GPU Out of Memory

If you encounter CUDA OOM errors:

1. **Reduce batch size**

   ```python
   per_device_train_batch_size=4  # Reduce from 8 to 4
   ```

2. **Use gradient accumulation**

   ```python
   gradient_accumulation_steps=2
   ```

3. **Enable mixed precision**

   ```python
   fp16=True  # Reduces memory by ~50%
   ```

4. **Clear GPU cache**

   ```python
   import torch
   torch.cuda.empty_cache()
   ```

### Authentication Issues with Hub

If `notebook_login()` doesn't work:

```bash
# Use CLI instead
huggingface-cli login
# Paste your token from https://huggingface.co/settings/tokens
```

Or pass token explicitly:

```python
from huggingface_hub import login
login(token="hf_xxxxxxxxxxxxx")
```

### Slow Training Speed

- Enable mixed precision (fp16=True)
- Increase `num_workers` in data loader
- Use GPU instead of CPU
- Reduce max_length from 256 to 128 if possible

## References

### Official Documentation

- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/)
- [BERT Model Card](https://huggingface.co/bert-base-uncased)
- [Hugging Face Hub](https://huggingface.co/)
- [Hugging Face Datasets Library](https://huggingface.co/docs/datasets/)

### Related Resources

- [IMDB Dataset](https://huggingface.co/datasets/imdb)
- [Fine-tuning Guide](https://huggingface.co/docs/transformers/training)
- [Trainer API Documentation](https://huggingface.co/docs/transformers/main_classes/trainer)
- [Model Card Template](https://huggingface.co/docs/hub/model-cards)

### Model on Hub

- **Hosted Model**: [Jim1892/IMDB-BERT-Finetuned](https://huggingface.co/Jim1892/IMDB-BERT-Finetuned)
- **IMDB Dataset**: [IMDB on Hugging Face Datasets](https://huggingface.co/datasets/imdb)

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) file for details.

## Author

Created for fine-tuning and deploying BERT models to Hugging Face Hub.

For questions or issues, please open a GitHub issue in this repository.
