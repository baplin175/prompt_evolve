"""ModelJudgeScorer — rubric-based LLM scoring."""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from app.gateway.base import GatewayClient
from app.models.run_result import RunResult
from app.models.score import ScoreBreakdown

logger = logging.getLogger(__name__)

_JUDGE_SYSTEM_PROMPT = """You are a rigorous, fair evaluator of AI model outputs.
You will be given a task description, the model output, and evaluation criteria.
Score the output on each criterion from 0.0 to 1.0 and provide brief reasoning.
Respond ONLY with valid JSON in this exact format:
{
  "criteria_scores": {
    "<criterion_name>": {"score": <float 0-1>, "reasoning": "<brief>"}
  },
  "overall_score": <float 0-1>,
  "summary": "<one sentence overall assessment>"
}"""

_JUDGE_USER_TEMPLATE = """TASK DESCRIPTION:
{task_description}

MODEL OUTPUT:
{model_output}

EVALUATION CRITERIA (with weights):
{criteria_text}

Score the output strictly based on the criteria above."""


class ModelJudgeScorer:
    """Rubric-based LLM scoring using a judge model."""

    name = "model_judge"
    dimension = "correctness"

    def __init__(self, gateway: GatewayClient, judge_model: Optional[str] = None) -> None:
        self._gateway = gateway
        self._judge_model = judge_model or "gpt-4o-mini"

    def score(
        self,
        result: RunResult,
        expected_output: str | None,
        config: dict[str, Any],
    ) -> ScoreBreakdown:
        """Score using an LLM judge.

        Config keys:
        - ``task_description``: What the prompt was asked to do.
        - ``criteria``: dict mapping criterion_name → {weight, description}.
        - ``weight``: Scorer weight in the aggregate (default 1.0).
        - ``judge_model``: Override the judge model (default from __init__).
        """
        weight = config.get("weight", 1.0)
        task_description = config.get("task_description", "Complete the user's request accurately.")
        criteria: dict[str, Any] = config.get("criteria", {
            "accuracy": {"weight": 0.5, "description": "Is the output factually correct and complete?"},
            "clarity": {"weight": 0.3, "description": "Is the output clear and well-structured?"},
            "relevance": {"weight": 0.2, "description": "Is the output relevant to the task?"},
        })
        judge_model = config.get("judge_model", self._judge_model)

        criteria_text = "\n".join(
            f"- {name} (weight={info.get('weight', 1.0)}): {info.get('description', '')}"
            for name, info in criteria.items()
        )

        if expected_output:
            task_description += f"\n\nExpected output (for reference): {expected_output}"

        user_content = _JUDGE_USER_TEMPLATE.format(
            task_description=task_description,
            model_output=result.raw_output or "(empty output)",
            criteria_text=criteria_text,
        )

        try:
            response = self._gateway.complete(
                model=judge_model,
                temperature=0.0,
                max_tokens=512,
                system_prompt=_JUDGE_SYSTEM_PROMPT,
                user_content=user_content,
            )
            raw = response.content.strip()
            # Strip markdown fences
            if raw.startswith("```"):
                lines = raw.splitlines()
                raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

            parsed = json.loads(raw)
            overall = float(parsed.get("overall_score", 0.0))
            overall = max(0.0, min(1.0, overall))
            details = {
                "criteria_scores": parsed.get("criteria_scores", {}),
                "summary": parsed.get("summary", ""),
                "judge_model": judge_model,
            }
        except json.JSONDecodeError as exc:
            logger.warning("ModelJudgeScorer: failed to parse judge response: %s", exc)
            overall = 0.0
            details = {"error": f"JSON parse error: {exc}", "raw": response.content if 'response' in dir() else ""}
        except Exception as exc:
            logger.error("ModelJudgeScorer: unexpected error: %s", exc)
            overall = 0.0
            details = {"error": str(exc)}

        return ScoreBreakdown(
            run_result_id=result.id,
            scorer_name=self.name,
            dimension=self.dimension,
            raw_score=overall,
            weight=weight,
            weighted_score=overall * weight,
            details=details,
        )
