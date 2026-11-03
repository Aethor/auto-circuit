"""Offline, tiny GPT-2 TransformerBridge path-patching smoke test.

Run: uv run --no-sync python experiments/bridge_path_patching.py
No tokenizer or model checkpoint is downloaded.
"""
import torch
from transformers import GPT2Config, GPT2LMHeadModel
from transformer_lens.model_bridge import TransformerBridge

from auto_circuit.model_utils.transformer_lens_utils import enable_graph_hooks
from auto_circuit.utils.ablation_activations import src_ablations
from auto_circuit.utils.graph_utils import patch_mode, patchable_model


def main() -> None:
    torch.manual_seed(0)
    hf = GPT2LMHeadModel(
        GPT2Config(n_layer=2, n_head=2, n_embd=16, n_positions=16, vocab_size=64)
    ).eval()
    bridge = TransformerBridge.boot_transformers("gpt2", hf_model=hf, device="cpu")
    enable_graph_hooks(bridge)
    model = patchable_model(bridge, factorized=True, separate_qkv=True)
    clean = torch.tensor([[1, 2, 3, 4]])
    corrupt = torch.tensor([[1, 7, 8, 4]])
    with torch.no_grad():
        clean_logits = model(clean)
        corrupt_logits = model(corrupt)
        ablated = src_ablations(model, corrupt)
        edge = next(
            e for e in model.edges
            if e.src.name == "A0.0" and e.dest.name == "A1.0.Q"
        )
        with patch_mode(model, ablated, edges=[edge]):
            patched_logits = model(clean)
        restored_logits = model(clean)
    delta = (patched_logits - clean_logits).abs().max().item()
    print(f"Patched path: {edge.name}")
    print(f"Clean last-token argmax: {clean_logits[0, -1].argmax().item()}")
    print(f"Corrupt last-token argmax: {corrupt_logits[0, -1].argmax().item()}")
    print(f"Max |path-patched - clean|: {delta:.6g}")
    assert delta > 0
    assert torch.allclose(restored_logits, clean_logits, atol=1e-6)


if __name__ == "__main__":
    main()
