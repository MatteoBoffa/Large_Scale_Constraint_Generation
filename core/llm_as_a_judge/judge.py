from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any

from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams


@dataclass
class SemanticEquivalenceResult:
    passed: bool
    score: float
    reason: Optional[str] = None


def eval_semantic_equivalence_geval(
    gt: str,
    generated: str,
    *,
    judge_model: str = "gpt-4.1",
    threshold: float = 0.9,
) -> SemanticEquivalenceResult:
    """
    Returns whether `generated` is semantically equivalent to `gt` (meaning exactly the same thing).
    Uses G-Eval (LLM-as-a-judge) with strict evaluation steps.

    Requirements:
      - pip install -U deepeval openai
      - export OPENAI_API_KEY="..."
    """

    metric = GEval(
        name="SemanticEquivalenceExact",
        evaluation_steps=[
            "Interpret 'expected output' as the ground truth (GT).",
            "The actual output may replace forbidden words. Different wording is allowed.",
            "",
            "Goal: FUNCTIONAL SEMANTIC EQUIVALENCE.",
            "The generated sentence should preserve the same overall meaning, scene, intent, and key facts.",
            "Allow near-synonyms and category-level substitutions when they do NOT change what a reader would reasonably infer or do.",
            "",
            "Pass conditions (must satisfy all):",
            "1) No contradiction: the generated sentence must not contradict GT.",
            "2) No major additions/omissions: it must not add or remove important facts. Minor stylistic details are okay.",
            "3) Replacements must preserve ROLE/FUNCTION in context. If the replaced word is an object/tool, it should be suitable for the same action.",
            "",
            "Be permissive for everyday near-synonyms (e.g., bathroom/restroom; modify/change; shaving cream/shaving lotion) "
            "when they preserve the intended action and scene.",
            "",
            "Be strict for substitutions that change:",
            "- Names of specific event/holiday/occasion (e.g., 'Remembrance Day' changed to 'Tribute day').",
            "- Safety-critical or instruction-critical details (numbers, dosages, timing, negation, must/should, etc.).",
            "- Tool appropriateness or specificity in a way that changes the action (e.g., scrub brush -> broom for cleaning a toilet).",
            "",
            "Scoring guidance:",
            "1.0 = clearly equivalent (perfect paraphrase or safe near-synonym substitutions).",
            "0.8-0.95 = mostly equivalent with slight loss of specificity but still same intent/scene.",
            "0.5-0.8 = borderline; meaning plausibly shifts or becomes too vague.",
            "<0.5 = meaning changed materially.",
        ],
        evaluation_params=[
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        threshold=threshold,
        model=judge_model,
    )

    test_case = LLMTestCase(
        input="semantic equivalence check",
        actual_output=generated,
        expected_output=gt,
    )

    metric.measure(test_case)

    # DeepEval commonly exposes `score` and a textual `reason`. :contentReference[oaicite:2]{index=2}
    score = float(getattr(metric, "score", 0.0))
    passed = bool(getattr(metric, "is_successful", lambda: score >= threshold)())
    reason = getattr(metric, "reason", None) or getattr(metric, "explanation", None)

    return SemanticEquivalenceResult(passed=passed, score=score, reason=reason)
