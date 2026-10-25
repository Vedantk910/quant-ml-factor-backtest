---
name: pytorch-peft-mlops
description: Idiomatic PyTorch deep learning, LLM fine-tuning via Hugging Face PEFT (LoRA, QLoRA with bitsandbytes NF4), memory-efficient SDPA attention, and production MLOps data drift monitoring (PSI). Use when authoring neural network architectures or training pipelines.
allowed-tools: [Bash, Read, Edit]
compatibility: Python 3.10+, PyTorch >= 2.1, peft, transformers, bitsandbytes
---
# PyTorch, PEFT & Production MLOps Protocol

## Engineering Standards
1. Scaled Dot-Product Attention: Enforce `torch.nn.functional.scaled_dot_product_attention`.
2. Zero Grad: Always use `optimizer.zero_grad(set_to_none=True)` to minimize memory churn.
3. Quantization: QLoRA must use NF4 with double quantization and `torch.bfloat16` compute dtype.
4. Data Drift: Track Population Stability Index (PSI); flag PSI >= 0.25 as retraining triggers.

## Production Implementations

```python
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

def get_qlora_model(model_id: str, r: int = 16, alpha: int = 32):
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_cfg,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    base_model = prepare_model_for_kbit_training(base_model)
    peft_cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(base_model, peft_cfg)
```
