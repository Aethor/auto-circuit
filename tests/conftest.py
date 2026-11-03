from typing import Any, Optional

import pytest
import torch as t
from transformers import GPT2Config, GPT2LMHeadModel
from transformer_lens.model_bridge import TransformerBridge

from auto_circuit.model_utils.transformer_lens_utils import enable_graph_hooks

from auto_circuit.data import PromptDataLoader, load_datasets_from_json
from auto_circuit.model_utils.micro_model_utils import MicroModel
from auto_circuit.utils.misc import repo_path_to_abs_path

DEVICE = t.device("cuda") if t.cuda.is_available() else t.device("cpu")


def pytest_addoption(parser: Any) -> None:
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )


def pytest_configure(config: Any) -> None:
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(config: Any, items: Any):
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


@pytest.fixture(scope="session")
def mini_tl_transformer() -> TransformerBridge:
    hf = GPT2LMHeadModel(
        GPT2Config(n_layer=2, n_embd=16, n_head=2, n_positions=64, vocab_size=128)
    ).to(DEVICE).eval()
    model = TransformerBridge.boot_transformers(
        "gpt2", hf_model=hf, device=str(DEVICE)
    )
    enable_graph_hooks(model)
    model.eval()
    return model


def get_transformer(
    request: Optional[pytest.FixtureRequest], model_name: Optional[str] = None
) -> TransformerBridge:
    name = request.param if request is not None else None
    if name is None:
        assert model_name is not None
        name = model_name
    model = TransformerBridge.boot_transformers(name, device=str(DEVICE))
    enable_graph_hooks(model)
    for param in model.parameters():
        param.requires_grad = False
    return model


@pytest.fixture(scope="session")
def hooked_transformer(
    request: pytest.FixtureRequest, model_name: Optional[str] = None
) -> TransformerBridge:
    return get_transformer(request, model_name)


@pytest.fixture(scope="session")
def gpt2() -> TransformerBridge:
    return get_transformer(None, "gpt2")


@pytest.fixture(scope="session")
def micro_dataloader(
    multiple_answers: bool = False, batch_count: int = 1, batch_size: int = 1
) -> PromptDataLoader:
    dataloader_len = batch_size * batch_count
    file_name = f"micro_model_inputs{'_multiple_answers' if multiple_answers else ''}"
    _, test_loader = load_datasets_from_json(
        None,
        repo_path_to_abs_path(f"datasets/{file_name}.json"),
        device=DEVICE,
        prepend_bos=False,
        batch_size=batch_size,
        train_test_size=(dataloader_len, dataloader_len),
    )
    return test_loader


@pytest.fixture(scope="session")
def micro_model() -> MicroModel:
    return MicroModel(n_layers=2).to(DEVICE)
