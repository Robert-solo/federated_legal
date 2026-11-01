"""Independent additive shared/private LoRA branches for federated clients."""

from __future__ import annotations

import torch
from peft import get_peft_model_state_dict, set_peft_model_state_dict


class DualLora:
    """Keep private LoRA factors separate; only the default adapter is uploaded."""

    def __init__(self, model, config) -> None:
        self.model = model
        model.add_adapter("local", config)
        self.enabled = False
        self.activate(local=False, train="shared")

    def activate(self, *, local: bool, train: str | None = None) -> None:
        self.model.base_model.set_adapter(["default", "local"] if local else ["default"])
        for name, parameter in self.model.named_parameters():
            parameter.requires_grad = (
                train == "shared" and ".default." in name
            ) or (train == "local" and ".local." in name)

    def state(self) -> dict[str, torch.Tensor]:
        return {
            key: value.detach().cpu().clone()
            for key, value in get_peft_model_state_dict(
                self.model, adapter_name="local"
            ).items()
        }

    def load(self, state: dict[str, torch.Tensor]) -> None:
        set_peft_model_state_dict(self.model, state, adapter_name="local")

    def reset(self) -> None:
        # A remains initialized; zero B makes the branch output zero and trainable.
        for name, parameter in self.model.named_parameters():
            if "lora_B.local." in name:
                with torch.no_grad():
                    parameter.zero_()
        self.enabled = False
