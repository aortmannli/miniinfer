import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from miniinfer.model import Qwen3Config, Qwen3ForCausalLM, load_weights

MODEL_ID = "Qwen/Qwen3-0.6B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="session")
def tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_ID)


@pytest.fixture(scope="session")
def hf_model():
    # fp32 + eager attention: slowest, but the least numerical noise to hide bugs behind
    m = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float32, attn_implementation="eager"
    )
    return m.to(DEVICE).eval()


@pytest.fixture(scope="session")
def mini_model():
    cfg = Qwen3Config.from_hf(AutoConfig.from_pretrained(MODEL_ID))
    m = load_weights(Qwen3ForCausalLM(cfg), MODEL_ID)
    return m.float().to(DEVICE).eval()
