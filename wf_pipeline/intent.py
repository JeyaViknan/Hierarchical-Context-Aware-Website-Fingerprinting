from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class IntentRule:
    """
    A simple, explainable rule:
    if all required categories appear (order-agnostic) in the session window,
    emit an intent label.
    """

    name: str
    required_any_order: Tuple[str, ...]
    description: str


DEFAULT_RULES: List[IntentRule] = [
    IntentRule(
        name="Sports merchandising",
        required_any_order=("Sports", "Shopping"),
        description="User appears to browse sports content and shop for related items.",
    ),
    IntentRule(
        name="Current affairs study",
        required_any_order=("Education", "News"),
        description="User appears to combine learning resources with news reading.",
    ),
    IntentRule(
        name="Trip planning",
        required_any_order=("Travel", "Food"),
        description="User appears to plan travel while exploring food/restaurant options.",
    ),
]


def infer_intent(category_sequence: List[str], rules: Optional[List[IntentRule]] = None) -> Dict[str, str]:
    """
    Infer intent from a category sequence using transparent heuristics.

    Returns a dict with:
    - intent: short label
    - rationale: human-readable explanation
    """

    rules = DEFAULT_RULES if rules is None else rules
    seen = set(category_sequence)
    for r in rules:
        if all(req in seen for req in r.required_any_order):
            return {"intent": r.name, "rationale": r.description}
    return {
        "intent": "General browsing",
        "rationale": "No specific pattern matched the current intent rules.",
    }

