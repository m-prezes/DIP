import torch as t
from transformers import AutoModelForCausalLM, AutoTokenizer

HUGGINGFACE_TOKEN = "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"
LLAMA_2_13B_CHAT = "meta-llama/Llama-2-13b-chat-hf"
LLAMA_2_7B_CHAT = "meta-llama/Llama-2-7b-chat-hf"


def load_model_and_tokenizer(model: str = LLAMA_2_7B_CHAT):
    if model == "LLAMA_2_13B_CHAT":
        model = LLAMA_2_13B_CHAT
    elif model == "LLAMA_2_7B_CHAT":
        model = LLAMA_2_7B_CHAT
    else:
        raise NotImplementedError(
            f"{model} not available. Choose one from {[LLAMA_2_13B_CHAT, LLAMA_2_7B_CHAT]}"
        )

    tokenizer = AutoTokenizer.from_pretrained(model, use_auth_token=HUGGINGFACE_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(
        model, load_in_8bit=True, use_auth_token=HUGGINGFACE_TOKEN
    )

    return model, tokenizer
