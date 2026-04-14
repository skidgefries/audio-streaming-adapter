from __future__ import annotations

from typing import Any


class WandbLogger:
    def __init__(
        self,
        *,
        enabled: bool,
        project: str,
        entity: str | None,
        run_name: str | None,
        config: dict,
        terminal: bool = True,
        terminal_prefix: str = "metrics",
    ):
        self.enabled = enabled
        self._wandb = None
        self._run = None
        self._terminal = bool(terminal)
        self._terminal_prefix = str(terminal_prefix)
        if not enabled:
            return

        try:
            import wandb  # type: ignore
        except Exception:
            self.enabled = False
            return

        self._wandb = wandb
        self._run = wandb.init(project=project, entity=entity, name=run_name, config=config)

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        # Terminal logging (works even if wandb is disabled/not installed)
        if self._terminal:
            parts: list[str] = []
            for k in sorted(metrics.keys()):
                v = metrics[k]
                if isinstance(v, float):
                    parts.append(f"{k}={v:.4f}")
                else:
                    parts.append(f"{k}={v}")
            step_str = f" step={step}" if step is not None else ""
            print(f"[{self._terminal_prefix}]{step_str} " + " ".join(parts))

        # W&B logging (if enabled)
        if self.enabled and self._wandb is not None:
            self._wandb.log(metrics, step=step)

    def finish(self) -> None:
        if not self.enabled or self._run is None:
            return
        self._run.finish()
