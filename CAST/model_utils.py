import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def create_bnb_config():
    return BitsAndBytesConfig(
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=False
    )


def load_model(model_name, bnb_config):
    n_gpus     = torch.cuda.device_count()
    max_memory = "20000MB"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        max_memory={i: max_memory for i in range(n_gpus)},
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=True)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer
