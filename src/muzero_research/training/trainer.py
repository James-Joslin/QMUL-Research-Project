"""MuZero unrolled training loss, implemented as an ordinary Python object."""

import numpy as np
import torch

from ..decision.common import model_device
from ..networks.models import scalar_to_support, support_to_scalar


def scale_gradient(tensor, scale):
    return tensor.detach() + (tensor - tensor.detach()) * scale


class Trainer:
    def __init__(self, model, config):
        self.model, self.config = model, config
        self.step = 0
        t = config.training
        options = {"lr": t.learning_rate, "weight_decay": t.weight_decay}
        if t.optimizer == "SGD":
            options["momentum"] = t.momentum
        self.optimizer = getattr(torch.optim, t.optimizer)(model.parameters(), **options)

    def update(self, batch):
        self.model.train()
        t = self.config.training
        device = model_device(self.model)
        tensors = {
            name: torch.as_tensor(value, device=device)
            for name, value in batch.items()
            if name != "indices"
        }
        targets_value = scalar_to_support(tensors["values"], self.config.network.support_size)
        targets_reward = scalar_to_support(tensors["rewards"], self.config.network.support_size)
        for group in self.optimizer.param_groups:
            group["lr"] = t.learning_rate * t.lr_decay_rate ** (self.step / t.lr_decay_steps)
        values, rewards, logits, state = self.model.initial_inference(tensors["observations"])
        losses = {"value": [], "reward": [], "policy": []}
        priorities = []
        total = torch.zeros(len(values), device=device)
        for unroll in range(t.unroll_steps + 1):
            if unroll:
                values, rewards, logits, state = self.model.recurrent_inference(
                    state, tensors["actions"][:, unroll, None]
                )
                state = scale_gradient(state, 0.5)
            value_loss = (
                -(targets_value[:, unroll] * torch.log_softmax(values, dim=1)).sum(dim=1)
                * tensors["masks"][:, unroll]
            )
            reward_loss = (
                -(targets_reward[:, unroll] * torch.log_softmax(rewards, dim=1)).sum(dim=1)
                * tensors["masks"][:, unroll]
                if unroll
                else torch.zeros_like(value_loss)
            )
            policy_loss = (
                -(tensors["policies"][:, unroll] * torch.log_softmax(logits, dim=1)).sum(dim=1)
                * tensors["policy_masks"][:, unroll]
            )
            loss = t.value_loss_weight * value_loss + reward_loss + policy_loss
            if unroll:
                loss = scale_gradient(loss, 1 / tensors["scales"])
            total += loss
            for name, value in (
                ("value", value_loss),
                ("reward", reward_loss),
                ("policy", policy_loss),
            ):
                losses[name].append(value.detach())
            predicted = support_to_scalar(values, self.config.network.support_size).squeeze(-1)
            priorities.append(
                (predicted.detach() - tensors["values"][:, unroll])
                .abs()
                .pow(t.priority_alpha)
                .cpu()
                .numpy()
            )
        total = (total * tensors["weights"]).mean()
        if not torch.isfinite(total):
            raise FloatingPointError("MuZero training loss is not finite")
        self.optimizer.zero_grad(set_to_none=True)
        total.backward()
        if any(
            parameter.grad is not None and not torch.isfinite(parameter.grad).all()
            for parameter in self.model.parameters()
        ):
            raise FloatingPointError("MuZero training gradient is not finite")
        self.optimizer.step()
        self.step += 1
        metrics = {
            "training_step": self.step,
            "total_loss": total.item(),
            "learning_rate": self.optimizer.param_groups[0]["lr"],
            **{
                f"{name}_loss": torch.stack(value).sum(dim=0).mean().item()
                for name, value in losses.items()
            },
        }
        return np.stack(priorities, axis=1), metrics

    def state_dict(self):
        return {"step": self.step, "optimizer": self.optimizer.state_dict()}

    def load_state_dict(self, state):
        self.step = state["step"]
        if state["optimizer"] is not None:
            self.optimizer.load_state_dict(state["optimizer"])
