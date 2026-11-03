"""Tracr programs and TransformerLens 4 bridge conversion.

Tracr native bridges are useful for forward analysis, but their attention fork does
not support the per-head graph hooks used for path patching in this project.
"""
from typing import Any, Literal, Set, Tuple

import torch
from tracr.compiler import compiling
from tracr.compiler.assemble import AssembledTransformerModel
from tracr.compiler.lib import make_frac_prevs
from tracr.rasp import rasp
from transformer_lens.model_bridge import TransformerBridge
from transformer_lens.utilities.tracr import (
    make_tracr_transformer_bridge_config,
    make_tracr_transformer_bridge_state_dict,
)

BOS = "BOS"
REVERSE_VOCAB: Set[Any] = {1, 2, 3}
XPROPORTION_VOCAB: Set[Any] = {"w", "x", "y", "z"}
MAX_SEQ_LEN = 5
TRACR_TASK_KEY = Literal["reverse", "xproportion"]


def get_tracr_model(
    tracr_task_key: TRACR_TASK_KEY, device: str
) -> Tuple[TransformerBridge, AssembledTransformerModel]:
    """Compile a Tracr program and load categorical weights into a native bridge.

    Numeric Tracr outputs and per-head graph patching are not supported here.
    """
    if tracr_task_key == "reverse":
        length = rasp.SelectorWidth(
            rasp.Select(rasp.tokens, rasp.tokens, rasp.Comparison.TRUE)
        )
        opp_index = length - rasp.indices - 1
        reverse = rasp.Aggregate(
            rasp.Select(rasp.indices, opp_index, rasp.Comparison.EQ), rasp.tokens
        )
        program, vocab = reverse, REVERSE_VOCAB
    elif tracr_task_key == "xproportion":
        raise NotImplementedError(
            "Numeric Tracr outputs are not supported by the TL4 native converter"
        )
    else:
        raise ValueError(f"Unknown task {tracr_task_key}")

    tracr_model = compiling.compile_rasp_to_model(
        program, vocab=vocab, max_seq_len=MAX_SEQ_LEN, compiler_bos=BOS
    )
    config = make_tracr_transformer_bridge_config(tracr_model)
    bridge = TransformerBridge.boot_native(config, device=device)
    state = make_tracr_transformer_bridge_state_dict(tracr_model)
    # The converter omits bridge aliases for tied MLP/head parameters. Those are
    # initialized by boot_native; test output parity before relying on this loader.
    bridge.load_state_dict(state, strict=False)
    bridge.cfg.model_name = f"tracr-{tracr_task_key}"
    bridge.eval()
    return bridge, tracr_model
