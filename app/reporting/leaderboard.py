"""Leaderboard builder — ranks candidates by their aggregate score."""

from __future__ import annotations

from app.models.candidate import PromptCandidate
from app.models.score import CandidateScore


def build_leaderboard(
    scores: list[CandidateScore],
    candidates: list[PromptCandidate],
) -> list[dict]:
    """Merge candidate metadata with scores and return ranked leaderboard.

    Args:
        scores: Scored candidates (unsorted).
        candidates: All candidate objects.

    Returns:
        List of dicts ordered by aggregate_score descending.
    """
    cand_map = {c.id: c for c in candidates}
    ranked = sorted(scores, key=lambda s: s.aggregate_score, reverse=True)

    leaderboard = []
    for rank, score in enumerate(ranked, start=1):
        cand = cand_map.get(score.candidate_id)
        entry = {
            "rank": rank,
            "candidate_id": score.candidate_id,
            "aggregate_score": round(score.aggregate_score, 4),
            "dimension_scores": {k: round(v, 4) for k, v in score.dimension_scores.items()},
            "eval_case_count": score.eval_case_count,
            "error_count": score.error_count,
            "avg_latency_ms": round(score.avg_latency_ms, 1),
            "total_cost_usd": score.total_cost_usd,
            "mutation_strategy": cand.mutation_strategy if cand else "?",
            "parent_id": cand.parent_id if cand else None,
            "prompt_text_preview": (cand.prompt_text[:120] + "…") if cand and len(cand.prompt_text) > 120 else (cand.prompt_text if cand else "?"),
        }
        leaderboard.append(entry)

    return leaderboard
