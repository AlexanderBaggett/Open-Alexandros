from __future__ import annotations

import tempfile

import torch

import _bootstrap  # noqa: F401
from alexandros import AlexandrosForDiffusionLM
from alexandros.configuration_alexandros import load_config_file

config = load_config_file("configs/heavy_debug.yaml")
model = AlexandrosForDiffusionLM(config).eval()
input_ids = torch.tensor([[config.bos_token_id, 4, 5]], dtype=torch.long)
with tempfile.TemporaryDirectory() as tmpdir:
    model.save_pretrained(tmpdir)
    restored = AlexandrosForDiffusionLM.from_pretrained(tmpdir).eval()
    with torch.no_grad():
        original = model(input_ids).logits
        loaded = restored(input_ids).logits
print({"roundtrip": bool(torch.allclose(original, loaded, atol=1e-6))})
