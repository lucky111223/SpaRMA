from dataclasses import dataclass
from .training import SpaRMAConfig


@dataclass(frozen=True)
class DatasetPreset:
    radius: float
    batch_key: str
    training: SpaRMAConfig


_PRESETS = {
    "dlpfc4": DatasetPreset(150.0, "batch_name", SpaRMAConfig(positive_k=3, mnn_k=50, base_margin=1.0, seed=0)),
    "dlpfc12": DatasetPreset(150.0, "batch_name", SpaRMAConfig(positive_k=15, mnn_k=100, base_margin=1.0, seed=0)),
    "mouse_embryo": DatasetPreset(1.3, "batch_name", SpaRMAConfig(positive_k=15, mnn_k=100, base_margin=2.5, seed=42)),
}


def get_preset(name: str) -> DatasetPreset:
    try:
        return _PRESETS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown preset: {name}") from exc
