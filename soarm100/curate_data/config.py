from dataclasses import dataclass
from pathlib import Path

from lerobot.common.optim import OptimizerConfig
from lerobot.common.utils.hub import HubMixin
from lerobot.configs.default import DatasetConfig
from lerobot.configs.policies import PreTrainedConfig

@dataclass
class CurateDataConfig(HubMixin):
    dataset: DatasetConfig
    output_dir: Path | None = None
    seed: int | None = 1000
    num_workers: int = 4
    batch_size: int = 100
    optimizer: OptimizerConfig | None = None
    steps: int = 100_000
    save_freq: int = 20_000
    policy: PreTrainedConfig | None = None
    type: str = "states" # "states" or "actions"
