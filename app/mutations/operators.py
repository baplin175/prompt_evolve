"""All concrete mutation operator implementations."""

from __future__ import annotations

import logging
import random
from typing import Optional

from app.gateway.base import GatewayClient
from app.models.candidate import PromptCandidate

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Meta-prompt templates used for LLM-based mutations
# ---------------------------------------------------------------------------

_META_PROMPT_SYSTEM = (
    "You are an expert prompt engineer. Your task is to improve a given prompt "
    "by applying a specific mutation strategy. Return ONLY the improved prompt "
    "text — no preamble, no explanation, no markdown fencing."
)

_META_PROMPTS: dict[str, str] = {
    "simplify_wording": (
        "Rewrite the following prompt using simpler, clearer wording. "
        "Remove jargon and unnecessary complexity. Keep all original instructions intact.\n\n"
        "ORIGINAL PROMPT:\n{prompt_text}"
    ),
    "tighten_constraints": (
        "Rewrite the following prompt to add tighter, more explicit constraints and requirements. "
        "Make edge-case handling more precise. Do not change the core task.\n\n"
        "ORIGINAL PROMPT:\n{prompt_text}"
    ),
    "improve_formatting_instructions": (
        "Rewrite the following prompt to include clearer output format instructions. "
        "Add explicit format rules (e.g., bullet points, JSON structure, headings) "
        "that the model should follow. Do not change the core task.\n\n"
        "ORIGINAL PROMPT:\n{prompt_text}"
    ),
    "add_examples": (
        "Rewrite the following prompt to include 1–2 concrete examples that illustrate "
        "what a good response looks like. Use clear 'Input:' and 'Output:' labels. "
        "Do not change the core task.\n\n"
        "ORIGINAL PROMPT:\n{prompt_text}"
    ),
    "handle_ambiguity": (
        "Rewrite the following prompt to explicitly address potential ambiguities. "
        "Add clarifying instructions for edge cases or underspecified situations. "
        "Do not change the core task.\n\n"
        "ORIGINAL PROMPT:\n{prompt_text}"
    ),
    "reduce_verbosity": (
        "Rewrite the following prompt to be more concise. Remove redundant phrases, "
        "repetitions, and unnecessary words while preserving all essential instructions.\n\n"
        "ORIGINAL PROMPT:\n{prompt_text}"
    ),
    "reorder_instruction_blocks": (
        "Rewrite the following prompt by reordering the instruction blocks for better "
        "logical flow (e.g., context first, then task, then constraints, then format). "
        "Do not add or remove any instructions.\n\n"
        "ORIGINAL PROMPT:\n{prompt_text}"
    ),
}


def _llm_mutate(
    candidate: PromptCandidate,
    gateway: GatewayClient,
    strategy: str,
    meta_prompt: str,
    max_retries: int = 3,
) -> PromptCandidate:
    """Apply an LLM-based mutation strategy and return a new PromptCandidate."""
    user_content = meta_prompt.format(prompt_text=candidate.prompt_text)

    last_error: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            response = gateway.complete(
                model=candidate.model,
                temperature=0.7,
                max_tokens=2048,
                system_prompt=_META_PROMPT_SYSTEM,
                user_content=user_content,
            )
            new_text = response.content.strip()
            if not new_text:
                raise ValueError("Mutation produced empty prompt text")
            return PromptCandidate(
                parent_id=candidate.id,
                prompt_text=new_text,
                system_prompt=candidate.system_prompt,
                model=candidate.model,
                temperature=candidate.temperature,
                mutation_strategy=strategy,
                notes=f"Mutated from {candidate.id[:8]} via {strategy}",
            )
        except Exception as exc:
            last_error = exc
            logger.warning("Mutation attempt %d/%d failed: %s", attempt, max_retries, exc)

    raise RuntimeError(
        f"Mutation '{strategy}' failed after {max_retries} attempts: {last_error}"
    )


# ---------------------------------------------------------------------------
# Concrete operator classes
# ---------------------------------------------------------------------------


class SimplifyWording:
    name = "simplify_wording"

    def mutate(self, candidate: PromptCandidate, gateway: GatewayClient) -> PromptCandidate:
        return _llm_mutate(candidate, gateway, self.name, _META_PROMPTS[self.name])


class TightenConstraints:
    name = "tighten_constraints"

    def mutate(self, candidate: PromptCandidate, gateway: GatewayClient) -> PromptCandidate:
        return _llm_mutate(candidate, gateway, self.name, _META_PROMPTS[self.name])


class ImproveFormattingInstructions:
    name = "improve_formatting_instructions"

    def mutate(self, candidate: PromptCandidate, gateway: GatewayClient) -> PromptCandidate:
        return _llm_mutate(candidate, gateway, self.name, _META_PROMPTS[self.name])


class AddExamples:
    name = "add_examples"

    def mutate(self, candidate: PromptCandidate, gateway: GatewayClient) -> PromptCandidate:
        return _llm_mutate(candidate, gateway, self.name, _META_PROMPTS[self.name])


class HandleAmbiguity:
    name = "handle_ambiguity"

    def mutate(self, candidate: PromptCandidate, gateway: GatewayClient) -> PromptCandidate:
        return _llm_mutate(candidate, gateway, self.name, _META_PROMPTS[self.name])


class ReduceVerbosity:
    name = "reduce_verbosity"

    def mutate(self, candidate: PromptCandidate, gateway: GatewayClient) -> PromptCandidate:
        return _llm_mutate(candidate, gateway, self.name, _META_PROMPTS[self.name])


class ReorderInstructionBlocks:
    name = "reorder_instruction_blocks"

    def mutate(self, candidate: PromptCandidate, gateway: GatewayClient) -> PromptCandidate:
        return _llm_mutate(candidate, gateway, self.name, _META_PROMPTS[self.name])


class VaryModelParameters:
    """Mutate model parameters (temperature) without an LLM call."""

    name = "vary_model_parameters"

    def __init__(self, rng: Optional[random.Random] = None) -> None:
        self._rng = rng or random.Random()

    def mutate(self, candidate: PromptCandidate, gateway: GatewayClient) -> PromptCandidate:  # noqa: ARG002
        """Produce a copy of the candidate with a perturbed temperature."""
        delta = self._rng.uniform(-0.2, 0.2)
        new_temp = max(0.0, min(2.0, round(candidate.temperature + delta, 2)))
        return PromptCandidate(
            parent_id=candidate.id,
            prompt_text=candidate.prompt_text,
            system_prompt=candidate.system_prompt,
            model=candidate.model,
            temperature=new_temp,
            mutation_strategy=self.name,
            notes=(
                f"Mutated from {candidate.id[:8]} via {self.name}: "
                f"temp {candidate.temperature} → {new_temp}"
            ),
        )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ALL_OPERATORS: dict[str, object] = {
    "simplify_wording": SimplifyWording(),
    "tighten_constraints": TightenConstraints(),
    "improve_formatting_instructions": ImproveFormattingInstructions(),
    "add_examples": AddExamples(),
    "handle_ambiguity": HandleAmbiguity(),
    "reduce_verbosity": ReduceVerbosity(),
    "reorder_instruction_blocks": ReorderInstructionBlocks(),
    "vary_model_parameters": VaryModelParameters(),
}
