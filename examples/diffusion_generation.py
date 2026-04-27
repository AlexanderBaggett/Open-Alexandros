from __future__ import annotations

import torch

import _bootstrap  # noqa: F401
from alexandros import AlexandrosForDiffusionLM, GenerationMode
from alexandros.configuration_alexandros import load_config_file

config = load_config_file("configs/heavy_debug.yaml")
model = AlexandrosForDiffusionLM(config).eval()
prompt = torch.tensor([[config.bos_token_id, 4, 5]], dtype=torch.long)
with torch.no_grad():
    generated = model.generate(
        prompt,
        max_new_tokens=3,
        mode=GenerationMode.BLOCK_DIFFUSION,
    )
print(generated.tolist())
