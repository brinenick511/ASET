# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""ASET routing and sparse OLMoE expert dispatch for inference.

The OLMoE forward is adapted from calibration/models/debugging_olmoe.py
and Hugging Face Transformers 4.51.3. The selector follows ASET Algorithm 1.
"""

from dataclasses import dataclass
from types import MethodType

import torch
from torch.nn import functional as F
from transformers.models.olmoe.modeling_olmoe import OlmoeSparseMoeBlock


def select_aset_experts(
    router_logits: torch.Tensor, training_top_k: int, theta_min: float, theta_max: float, gamma: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return training top-k probabilities, expert IDs and the ASET retention mask."""
    expert_probs = F.softmax(router_logits, dim=-1, dtype=torch.float32)
    sorted_probs, sorted_expert_ids = expert_probs.sort(dim=-1, descending=True)
    cumulative_mass = sorted_probs.cumsum(dim=-1)
    sorted_positions = torch.arange(sorted_probs.size(-1), device=router_logits.device)

    # The paper's experiments use theta_cand = theta_max.
    candidate_count = (cumulative_mass < theta_max).sum(dim=-1, keepdim=True) + 1
    candidate_count = candidate_count.clamp_max(sorted_probs.size(-1))
    candidate_mass = cumulative_mass.gather(-1, candidate_count - 1)
    candidate_distribution = (
        sorted_probs.masked_fill(sorted_positions >= candidate_count, 0) / candidate_mass
    )

    candidate_entropy = -torch.special.xlogy(candidate_distribution, candidate_distribution).sum(
        dim=-1, keepdim=True
    )
    # A one-expert candidate has zero entropy. Clamp its denominator away from zero.
    entropy_denominator = candidate_count.clamp_min(2).to(candidate_entropy.dtype).log()
    normalized_entropy = (candidate_entropy / entropy_denominator).clamp(0, 1)
    dynamic_threshold = theta_min + (theta_max - theta_min) * normalized_entropy.pow(gamma)

    # Scale by the training-time top-k mass and retain the crossing expert.
    top_k_mass = cumulative_mass[:, training_top_k - 1 : training_top_k]
    retained_expert_count = (
        cumulative_mass[:, :training_top_k] < top_k_mass * dynamic_threshold
    ).sum(
        dim=-1, keepdim=True
    ) + 1
    retained_mask = sorted_positions[:training_top_k] < retained_expert_count
    return sorted_probs[:, :training_top_k], sorted_expert_ids[:, :training_top_k], retained_mask


def select_experts(
    router_logits: torch.Tensor, top_k: int, tau_min: float, tau_max: float, beta: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select ASET experts using the archived parameter names."""
    return select_aset_experts(router_logits, top_k, tau_min, tau_max, beta)


@dataclass
class RoutingStats:
    """Count routed positions including padding and retained token–expert pairs."""

    num_tokens: int = 0
    num_experts_activated: torch.Tensor | None = None


def install_aset(
    model: torch.nn.Module, tau_min: float, tau_max: float, beta: float
) -> list[RoutingStats]:
    """Install ASET on initialized OLMoE blocks and return their live routing counters."""
    if model.config.model_type != "olmoe":
        raise ValueError("ASET requires an OLMoE checkpoint")
    blocks = [module for module in model.modules() if isinstance(module, OlmoeSparseMoeBlock)]
    if len(blocks) != model.config.num_hidden_layers:
        raise RuntimeError("Unexpected OLMoE block layout; use Transformers 4.51.3")

    def aset_forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        token_states = hidden_states.reshape(-1, hidden_dim)
        router_logits = self.gate(token_states)
        routing_weights, top_k_expert_ids, retained_mask = select_aset_experts(
            router_logits,
            training_top_k=self.top_k,
            theta_min=tau_min,
            theta_max=tau_max,
            gamma=beta,
        )
        routing_weights = routing_weights.masked_fill(~retained_mask, 0)
        if self.norm_topk_prob:
            routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(token_states.dtype)

        layer_stats = self.aset_stats
        layer_stats.num_tokens += token_states.size(0)
        batch_expert_activations = retained_mask.sum()
        if layer_stats.num_experts_activated is None:
            layer_stats.num_experts_activated = batch_expert_activations
        else:
            layer_stats.num_experts_activated += batch_expert_activations

        final_token_states = torch.zeros_like(token_states)
        # Dispatch mask axes are expert, routing slot, then token.
        dispatch_mask = (
            F.one_hot(top_k_expert_ids, num_classes=self.num_experts).bool()
            & retained_mask.unsqueeze(-1)
        ).permute(2, 1, 0)
        for expert_id, expert in enumerate(self.experts):
            routing_slots, token_ids = torch.where(dispatch_mask[expert_id])
            if token_ids.numel() == 0:
                continue
            weighted_expert_output = (
                expert(token_states[token_ids]) * routing_weights[token_ids, routing_slots, None]
            )
            final_token_states.index_add_(0, token_ids, weighted_expert_output.to(token_states.dtype))
        return final_token_states.reshape(batch_size, sequence_length, hidden_dim), router_logits

    for block in blocks:
        block.aset_stats = RoutingStats()
        block.forward = MethodType(aset_forward, block)
    return [block.aset_stats for block in blocks]
