from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def _module():
    path = Path(__file__).resolve().parents[1] / "examples" / "convert_jax_model_to_pytorch.py"
    spec = importlib.util.spec_from_file_location("convert_jax_model_to_pytorch_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_mapped_and_presave_dtype_guards():
    mod = _module()
    mod._assert_float_dtype({"x": torch.ones(2, dtype=torch.float32)}, torch.float32, stage="mapped")
    with pytest.raises(ValueError, match="mapped"):
        mod._assert_float_dtype({"x": torch.ones(2, dtype=torch.bfloat16)}, torch.float32, stage="mapped")


def test_pi05_source_key_policy_is_exact():
    mod = _module()
    ok = SimpleNamespace(missing_keys=sorted(mod._PI05_EXPECTED_MISSING), unexpected_keys=[])
    mod._validate_source_keys(ok, pi05=True)
    with pytest.raises(ValueError, match="source load key mismatch"):
        mod._validate_source_keys(SimpleNamespace(missing_keys=[], unexpected_keys=[]), pi05=True)
    with pytest.raises(ValueError, match="source load key mismatch"):
        mod._validate_source_keys(
            SimpleNamespace(missing_keys=sorted(mod._PI05_EXPECTED_MISSING), unexpected_keys=["bogus"]),
            pi05=True,
        )


def test_converter_source_uses_explicit_pi05_and_preload_dtype():
    source = (Path(__file__).resolve().parents[1] / "examples" / "convert_jax_model_to_pytorch.py").read_text()
    assert 'dataclasses.replace(model_config, dtype=precision)' in source
    assert 'if "pi05" in checkpoint_dir' not in source
    assert 'pi0_model.load_state_dict(all_params, strict=False)' in source
    assert '_validate_source_keys(load_result, pi05=bool(model_config.pi05))' in source
    assert '_assert_float_dtype(all_params, torch.float32, stage="mapped Orbax tensors")' in source
    assert '_assert_float_dtype(pi0_model.state_dict(), torch_dtype, stage="pre-save model")' in source


def test_unknown_config_fails_before_conversion(monkeypatch):
    mod = _module()
    monkeypatch.setattr(mod._config, "get_config", lambda _: SimpleNamespace(name="fallback", model=object()))
    with pytest.raises(ValueError, match="not registered"):
        mod.main("/unused", "unknown", inspect_only=True)
