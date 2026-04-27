from __future__ import annotations

import torch

import _bootstrap  # noqa: F401
from alexandros import AlexandrosConfig, AlexandrosForCausalLM

config = AlexandrosConfig(
    vocab_size=64,
    hidden_size=32,
    intermediate_size=64,
    num_hidden_layers=2,
    num_attention_heads=4,
    max_position_embeddings=32,
    moe_num_experts=2,
    moe_top_k=1,
    moe_expert_hidden_size=24,
    kv_lora_rank=8,
    latent_dim=16,
    latent_slots=2,
    diffusion_steps=2,
    ttt_rank=4,
)
model = AlexandrosForCausalLM(config).eval()
input_ids = torch.tensor([[config.bos_token_id, 4, 5]], dtype=torch.long)
with torch.no_grad():
    output = model(input_ids)
print({"logits_shape": list(output.logits.shape)})
