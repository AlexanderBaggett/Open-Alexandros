# Evaluation Report: Alexandros <Checkpoint>

## Run Metadata

- Git commit:
- Config hash:
- Hardware:
- Python:
- Torch:
- Device and dtype:
- Tokenizer:

## Metrics

| Suite | Metric | Value | Notes |
| --- | ---: | ---: | --- |
| Smoke | Perplexity | | |
| Smoke | Diffusion reconstruction accuracy | | |
| Smoke | Latent reconstruction MSE | | Mechanism diagnostic, not reasoning quality |
| Smoke | Latent update norm | | |
| MoE | Mean load entropy | | Recent forward-pass expert balance |
| MoE | Timestep load entropy | | Per-denoising-step expert balance |
| MoE | Noisy-step load entropy | | Higher mask-probability timesteps |
| MoE | Polish-step load entropy | | Lower mask-probability timesteps |
| Long context | Needle target rank | | Synthetic token-rank diagnostic |
| Long context | Lost-middle worst rank | | Early/middle/late synthetic probes |
| Long context | Copy target rank | | Synthetic copy-token probe |
| Long context | Recurrent state max norm | | Cached SSM state diagnostic |
| Long context | Recurrent state mean update norm | | Cached SSM drift diagnostic |
| Reasoning | Toy modular-addition target rank | | Synthetic mechanics probe, not quality |
| Reasoning | Toy modular-addition target probability | | Synthetic mechanics probe |
| Runtime | Prefill ms | | |
| Runtime | Generation ms | | |
| Memory | Parameter bytes | | |

## Baselines

Describe any Heavy/Lite or external baseline comparisons and fair-tokenizer
caveats. Follow the policy in `docs/evaluation_strategy.md`.

## Ablations

List changed config fields, active parameters per token, tokenizer caveats,
hardware, precision, and command lines for any ablation.

## Known Gaps

Link open roadmap items that affect interpretation of results.
