<p align="center">
  <img src="https://i.ibb.co/5X8LK2TQ/ATHENA-PROJECT.png" alt="Athena Project" width="400"/>
</p>

<h1 align="center">EGen-Core</h1>

<p align="center">
  <strong>The Athena Project (2025–2026)</strong><br/>
  A milestone in efficient high-performance language modeling.<br/>
  Developed by <a href="https://github.com/ErebusTN">ErebusTN</a>.
</p>

<p align="center">
  <img src="https://cdn-avatars.huggingface.co/v1/production/uploads/66d6d5bf429249ec731ab9f1/yBoxFkum0Tm4Cv63yEmv8.png" alt="EGen-Core Logo" width="120"/>
</p>

<p align="center">
  <a href="https://pypi.org/project/EGen-Core/"><img src="https://img.shields.io/pypi/v/EGen-Core?color=blue&logo=pypi" alt="PyPI"/></a>
  <a href="https://github.com/ErebusTN/EGen-Core/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache-green.svg" alt="License"/></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python"/></a>
</p>

---

<div align="center">

[**Quick Start**](#-quick-start) |
[**Features**](#-features) |
[**Installation**](#-installation) |
[**Usage**](#-usage) |
[**Configuration**](#-configuration) |
[**API Reference**](#-api-reference) |
[**FAQ**](#-faq) |
[**Contributing**](#-contributing)

</div>

---

## 🌟 Overview

**EGen-Core** is a high-performance, memory-efficient inference engine for large language models. It enables running **70B+ parameter models on a single 4GB GPU** — without quantization, distillation, or pruning. With 8GB of VRAM, you can even run **405B Llama 3.1**.

The core technique is **layer-wise sharded inference**: the model is split into individual layer files and loaded one at a time during the forward pass. GPU memory is freed after each layer, keeping peak usage minimal. Optional **4-bit/8-bit block-wise compression** provides up to **3× inference speedup** with negligible accuracy loss.

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🧠 **Ultra-Low Memory Inference** | Run 70B models on 4GB VRAM, 405B on 8GB |
| ⚡ **3× Speed Boost** | Optional block-wise 4-bit/8-bit compression |
| 🔄 **Auto Model Detection** | `AutoModel.from_pretrained()` auto-detects architecture |
| 🏗️ **Multi-Architecture Support** | Llama, Mistral, Mixtral, QWen, ChatGLM, Baichuan, InternLM |
| 🍎 **Apple Silicon (MLX)** | Native macOS support via MLX acceleration |
| 📦 **Layer Prefetching** | Overlaps disk I/O with GPU compute for 10%+ speedup |
| 🔧 **HuggingFace Integration** | Direct model loading from HuggingFace Hub |

## 📦 Installation

### From PyPI

```bash
pip install EGen-Core
```

### With Compression Support

```bash
pip install EGen-Core[compression]
```

### For macOS (Apple Silicon)

```bash
pip install EGen-Core[mlx]
```

### From Source

```bash
git clone https://github.com/ErebusTN/EGen-Core.git
cd EGen-Core/egen_core
pip install -e .
```

## 🚀 Quick Start

```python
from egen_core import AutoModel

MAX_LENGTH = 128

# Load from HuggingFace Hub or local path
model = AutoModel.from_pretrained("garage-bAInd/Platypus2-70B-instruct")

input_text = ['What is the capital of United States?']

input_tokens = model.tokenizer(
    input_text,
    return_tensors="pt",
    return_attention_mask=False,
    truncation=True,
    max_length=MAX_LENGTH,
    padding=False
)

generation_output = model.generate(
    input_tokens['input_ids'].cuda(),
    max_new_tokens=20,
    use_cache=True,
    return_dict_in_generate=True
)

output = model.tokenizer.decode(generation_output.sequences[0])
print(output)
```

## 🔧 Configuration

When initializing a model, the following parameters are supported:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `compression` | `str` | `None` | `'4bit'` or `'8bit'` for block-wise quantization |
| `profiling_mode` | `bool` | `False` | Output layer loading time profiling |
| `layer_shards_saving_path` | `str` | `None` | Custom path for split model layers |
| `hf_token` | `str` | `None` | HuggingFace API token for gated models |
| `prefetching` | `bool` | `True` | Overlap model loading with compute |
| `delete_original` | `bool` | `False` | Delete original model after splitting to save disk |
| `max_seq_len` | `int` | `512` | Maximum sequence length |
| `device` | `str` | `"cuda:0"` | Target device |
| `dtype` | `torch.dtype` | `float16` | Model precision |

### Compression Example

```python
from egen_core import AutoModel

model = AutoModel.from_pretrained(
    "garage-bAInd/Platypus2-70B-instruct",
    compression='4bit'  # or '8bit'
)
```

### Gated Model Example

```python
model = AutoModel.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    hf_token='YOUR_HF_TOKEN'
)
```

## 🍎 macOS Support

EGen-Core supports Apple Silicon Macs via the [MLX](https://github.com/ml-explore/mlx) framework.

**Requirements:**
- Apple Silicon Mac (M1/M2/M3/M4)
- [MLX](https://github.com/ml-explore/mlx) installed
- Python native build (see [details](https://stackoverflow.com/a/65432861/21230266))

```python
from egen_core import AutoModel

model = AutoModel.from_pretrained("model-name")
output = model.generate(input_ids, max_new_tokens=20)
```

## 📖 API Reference

### `AutoModel.from_pretrained(model_name_or_path, **kwargs)`

Automatically detects model architecture and returns the appropriate model class.

**Supported Architectures:**

| Architecture | Class |
|-------------|-------|
| Llama / Llama 2 / Llama 3 | `EGenCoreLlama2` |
| Mistral | `EGenCoreMistral` |
| Mixtral | `EGenCoreMixtral` |
| QWen | `EGenCoreQWen` |
| QWen2 | `EGenCoreQWen2` |
| ChatGLM | `EGenCoreChatGLM` |
| Baichuan | `EGenCoreBaichuan` |
| InternLM | `EGenCoreInternLM` |

### Model Methods

| Method | Description |
|--------|-------------|
| `model.generate(input_ids, **kwargs)` | Generate text (inherits from `GenerationMixin`) |
| `model.forward(input_ids, **kwargs)` | Single forward pass with layer-wise loading |
| `model.tokenizer` | Access the model's tokenizer |

## ❓ FAQ

### 1. `MetadataIncompleteBuffer` Error
Likely caused by insufficient disk space. The model splitting process is disk-intensive. Clear your HuggingFace cache and ensure available space.

### 2. `ValueError: max() arg is an empty sequence`
Use `AutoModel` instead of a specific model class:
```python
from egen_core import AutoModel
model = AutoModel.from_pretrained("your-model")
```

### 3. `401 Client Error — Repo is gated`
Provide your HuggingFace token:
```python
model = AutoModel.from_pretrained("meta-llama/Llama-2-7b-hf", hf_token='YOUR_TOKEN')
```

### 4. `Asking to pad but tokenizer has no padding token`
Disable padding:
```python
input_tokens = model.tokenizer(input_text, padding=False, ...)
```

## 🤝 Contributing

Contributions, ideas, and discussions are welcome!

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the Apache License — see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <strong>The Athena Project</strong> — Developed by <a href="https://github.com/ErebusTN">ErebusTN</a> (2025–2026)
</p>