from pathlib import Path
import os

import numpy as np


def project_root() -> Path:
    return Path(os.environ.get("TR_ROOT", Path(__file__).resolve().parents[2])).resolve()


def data_root() -> Path:
    return Path(os.environ.get("TR_DATA_ROOT", project_root() / "dgl_dataset")).resolve()


def dataset_path(data_path: str, data: str) -> Path:
    base = Path(data_path)
    if not base.is_absolute():
        base = data_root() / data_path
    return (base / f"{data}.npz").resolve()


def load_graph_npz(data_path: str, data: str):
    path = dataset_path(data_path, data)
    # print(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Cannot find tr dataset '{data}' at {path}. "
            "Run scripts/prepare_suitesparse.py or set TR_DATA_ROOT."
        )
    return np.load(path)


def index_csv(default_name: str = "data_filter.csv") -> Path:
    explicit = os.environ.get("TR_DATA_CSV")
    if explicit:
        return Path(explicit).resolve()

    root = project_root()
    candidates = [root / default_name, root / "data_filter.csv", root / "data.csv"]
    for path in candidates:
        if path.exists():
            return path.resolve()
    return candidates[0].resolve()


def results_root() -> Path:
    root = Path(os.environ.get("TR_RESULTS_DIR", project_root() / "res")).resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def result_path(*parts: str) -> Path:
    path = results_root().joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
