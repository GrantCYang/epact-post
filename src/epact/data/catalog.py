"""Named original SFT conditions and their restored input locations."""

from dataclasses import dataclass
from pathlib import Path
import json
from research_common.paths import ROOT


@dataclass(frozen=True)
class ExperimentCatalog:
    data_root: Path
    conditions: dict
    seeds: tuple

    @classmethod
    def load(cls, data_root=None, config=None):
        value = json.loads(
            Path(config or ROOT / "configs/epact/experiments.json").read_text(
                encoding="utf-8"
            )
        )
        return cls(
            Path(data_root or ROOT / "data/epact").resolve(),
            value["conditions"],
            tuple(value["seeds"]),
        )

    @property
    def frozen(self):
        return self.data_root / "artifacts/balanced_eval_v4_frozen_v1"

    def experiment(self, name):
        return self.data_root / "experiments" / self.conditions[name]["directory"]

    def eval_root(self, name):
        return self.frozen if name == "full" else self.experiment(name) / "eval"
