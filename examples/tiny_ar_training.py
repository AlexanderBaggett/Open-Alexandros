from __future__ import annotations

import torch

import _bootstrap  # noqa: F401
from alexandros import AlexandrosForCausalLM
from alexandros.configuration_alexandros import load_config_file

config = load_config_file("configs/heavy_debug.yaml")
model = AlexandrosForCausalLM(config)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
input_ids = torch.tensor([[config.bos_token_id, 4, 5, 6]], dtype=torch.long)

for step in range(2):
    optimizer.zero_grad(set_to_none=True)
    output = model(input_ids, labels=input_ids)
    assert output.loss is not None
    output.loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    print({"step": step, "loss": round(float(output.loss.detach()), 4)})
