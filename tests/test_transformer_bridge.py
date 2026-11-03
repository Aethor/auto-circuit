"""Offline regression tests for the validated GPT-2 bridge graph."""
import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel
from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.model_bridge import TransformerBridge

from auto_circuit.model_utils.transformer_lens_utils import enable_graph_hooks
from auto_circuit.model_utils.sparse_autoencoders.autoencoder_transformer import sae_model
from auto_circuit.model_utils.task_projectors.projector_transformer import get_projector_model
from auto_circuit.utils.ablation_activations import src_ablations
from auto_circuit.utils.graph_utils import patch_mode, patchable_model


def tiny_bridge():
    torch.manual_seed(0)
    hf = GPT2LMHeadModel(
        GPT2Config(n_layer=2, n_head=2, n_embd=16, n_positions=16, vocab_size=64)
    ).eval()
    model = TransformerBridge.boot_transformers("gpt2", hf_model=hf, device="cpu")
    enable_graph_hooks(model)
    return model


def test_live_hookpoints_cache_and_path_patch():
    model = tiny_bridge()
    original_hook = model.hook_dict["blocks.0.attn.hook_result"]
    patched = patchable_model(model, factorized=True, separate_qkv=True)
    assert model.hook_dict["blocks.0.attn.hook_result"] is original_hook
    assert patched.n_edges > 0
    clean = torch.tensor([[1, 2, 3, 4]])
    corrupt = torch.tensor([[1, 7, 8, 4]])
    with torch.no_grad():
        baseline, cache = patched.run_with_cache(clean)
        for src in patched.srcs:
            assert src.module(patched) in patched.wrappers
            assert src.module_name in cache
        for dest in patched.dests:
            assert dest.module_name in cache
        assert cache["blocks.0.attn.hook_result"].shape == (1, 4, 2, 16)
        ablated = src_ablations(patched, corrupt)
        edge = next(
            e for e in patched.edges
            if e.src.name == "A0.0" and e.dest.name == "A1.0.Q"
        )
        with patch_mode(patched, ablated, edges=[edge]):
            altered = patched(clean)
        assert (altered - baseline).abs().max() > 1e-8
        assert torch.allclose(patched(clean), baseline, atol=1e-6)


def test_reject_native_attention_graph():
    native = TransformerBridge.boot_native(
        TransformerBridgeConfig(
            d_model=16, d_head=8, n_heads=2, n_layers=1,
            n_ctx=16, d_vocab=32, d_mlp=32,
        ), device="cpu"
    )
    with pytest.raises(NotImplementedError, match="GPT2LMHeadModel"):
        patchable_model(native, factorized=True, separate_qkv=True)
    with pytest.raises(NotImplementedError, match="GPT2LMHeadModel"):
        enable_graph_hooks(native)


def test_sae_and_projector_hooks_are_live():
    tokens = torch.tensor([[1, 2, 3]])
    bridge = tiny_bridge()
    baseline = bridge(tokens)
    sae = sae_model(
        bridge, sae_input="resid_delta_mlp", load_pretrained=False,
        n_latents=4, new_instance=False,
    )
    assert not torch.allclose(sae(tokens), baseline)
    with pytest.raises(NotImplementedError, match="SAE latent graph"):
        patchable_model(sae, factorized=True, separate_qkv=True)

    bridge = tiny_bridge()
    projector = get_projector_model(
        bridge, projector_input="resid_delta_mlp", mask_fn=None,
        load_pretrained=False, new_instance=False,
    )
    assert len(projector.projectors) == 2
    assert torch.allclose(projector(tokens), bridge(tokens), atol=1e-6)
    with pytest.raises(ValueError, match="existing blocks"):
        get_projector_model(
            bridge, projector_input="resid_delta_mlp", mask_fn=None,
            load_pretrained=False, new_instance=False, layer_idxs=[2],
        )
    with pytest.raises(NotImplementedError):
        patchable_model(projector, factorized=True, separate_qkv=True)


def test_prefix_cache_is_reused_without_mutating_stored_cache():
    model = tiny_bridge()
    prefix = torch.tensor([[1, 2]])
    suffix = torch.tensor([[3, 4]])
    with torch.inference_mode():
        _, kv = model(prefix, return_type="logits_and_cache")
    initial_len = kv.get_seq_length()
    patched = patchable_model(
        model, factorized=True, separate_qkv=True, kv_caches=(kv,)
    )
    with torch.no_grad():
        expected = model(torch.cat([prefix, suffix], dim=1))[:, 2:]
        actual, activations = patched.run_with_cache(suffix)
        assert "blocks.0.attn.hook_result" in activations
        assert torch.allclose(actual, expected, atol=1e-5)
        assert torch.allclose(patched(suffix), actual, atol=1e-5)
        assert kv.get_seq_length() == initial_len


def test_tracr_native_conversion_forward_and_graph_rejection():
    from auto_circuit.model_utils.tracr_model_utils import get_tracr_model

    bridge, tracr = get_tracr_model("reverse", "cpu")
    tokens = tracr.input_encoder.encode(["BOS", 3, 2, 1])
    expected = tracr.apply(["BOS", 3, 2, 1]).decoded[1:]
    decoded = tracr.output_encoder.decode(
        bridge(torch.tensor(tokens)).argmax(-1)[0, 1:].tolist()
    )
    assert decoded == expected
    with pytest.raises(NotImplementedError, match="GPT2LMHeadModel"):
        patchable_model(bridge, factorized=True, separate_qkv=True)
    with pytest.raises(NotImplementedError, match="Numeric Tracr"):
        get_tracr_model("xproportion", "cpu")
