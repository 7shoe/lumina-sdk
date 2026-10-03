"""Checkpoint class routing must distinguish RGAT from homogeneous GAT."""
from types import SimpleNamespace

import pytest


def load_example(monkeypatch, name):
    import importlib.util
    from pathlib import Path
    examples = Path(__file__).resolve().parents[2] / "example" / "opf"
    monkeypatch.syspath_prepend(str(examples))
    spec = importlib.util.spec_from_file_location("prerequisite_" + name, examples / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("class_path,homogeneous", [
    ("lumina.model.opf.hetero_model.RGAT", False),
    ("lumina.model.opf.hetero_model.OPFHeteroGNN", False),
    ("lumina.model.opf.hetero_model.HGT", False),
    ("lumina.model.opf.hetero_model.HEAT", False),
    ("lumina.model.opf.homo_model.GAT", True),
    ("lumina.model.opf.homo_model.GCN", True),
    ("lumina.model.opf.homo_model.GIN", True),
    ("lumina.model.opf.homo_model.TRANSFORMER", True),
])
def test_checkpoint_evaluation_dataset_kind(monkeypatch, class_path, homogeneous):
    module = load_example(monkeypatch, "evaluate_out_of_sample")
    monkeypatch.setattr("sys.argv", ["evaluate_out_of_sample.py", "--checkpoint-file",
                                    "unused.pt", "--test-cases", "case14", "--device", "cpu"])
    monkeypatch.setattr(module.Modeler, "load_model_from_training_checkpoint",
                        lambda *args, **kwargs: SimpleNamespace())
    monkeypatch.setattr(module.torch, "load", lambda *args, **kwargs: {"model_class": class_path})
    class DataDispatchReached(Exception):
        pass
    def capture(*args, **kwargs):
        assert kwargs["homogeneous"] is homogeneous
        raise DataDispatchReached
    monkeypatch.setattr(module, "load_test_datasets", capture)
    with pytest.raises(DataDispatchReached):
        module.main()

