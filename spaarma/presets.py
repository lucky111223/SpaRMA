from dataclasses import dataclass
from typing import Optional, Tuple
from .training import SpaRMAConfig


@dataclass(frozen=True)
class DatasetPreset:
    radius: float
    batch_key: str
    training: SpaRMAConfig
    alignment_key: Optional[str] = None
    batch_pairs: Optional[Tuple[Tuple[str, str], ...]] = None


_PRESETS = {
    "dlpfc4": DatasetPreset(150.0, "batch_name", SpaRMAConfig(positive_k=3, mnn_k=50, base_margin=1.0, seed=42)),
    "dlpfc12": DatasetPreset(150.0, "batch_name", SpaRMAConfig(positive_k=15, mnn_k=100, base_margin=1.0, seed=2025), alignment_key="sample_name"),
    "mouse_embryo": DatasetPreset(1.3, "batch_name", SpaRMAConfig(positive_k=15, mnn_k=100, base_margin=2.5, seed=42), batch_pairs=(("E9.5_E1S1", "E12.5_E1S1"), ("E10.5_E2S1", "E12.5_E1S1"), ("E11.5_E1S1", "E12.5_E1S1"))),
}


def get_preset(name: str) -> DatasetPreset:
    try:
        return _PRESETS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown preset: {name}") from exc
