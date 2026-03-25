# src/vision/classifier.py
"""
Rule-based classifier for the M&M sorting pipeline.

Runs all registered rules against a Features object, collects fired decisions,
and selects the winner by lowest priority then highest confidence. Stage 1 and 2
rejection rules always override stage 3 colour decisions by construction.

Usage:
    classifier = Classifier(cfg)
    decision = classifier.classify(features)
"""

import logging

from config import Config
from config.constants import ColourID
from src.vision import Features
from src.vision.rule import Decision, Rule, _RULE_REGISTRY

log = logging.getLogger(__name__)

class Classifier:
    """
    applies all active rules to a Features object and returns the winning Decision.

    rules are sorted by priority at construction time. all rules are evaluated
    for every input (no early exit) so that co-firing rules remain visible in
    debug logs during threshold tuning.
    """

    def __init__(self, cfg: Config, rules: list[Rule] | None = None) -> None:
        if rules is not None:
            self._rules = sorted(rules, key=lambda r: r.priority)
        else:
            import src.vision.rules  # noqa: F401
            self._rules = sorted(
                (cls(cfg) for cls in _RULE_REGISTRY),
                key=lambda r: r.priority,
            )
        log.info(
            "classifier initialised with %d rules: %s",
            len(self._rules),
            ", ".join(r.name for r in self._rules),
        )

    def classify(self, f: Features) -> Decision:
        """
        run all rules and return the winning Decision.

        selection key: lowest priority number, then highest confidence.
        if no rule fires, returns Decision(NON_MM, 0.0, "none", 999) as a
        fail-safe reject rather than propagating a None label downstream.
        """
        decisions: list[Decision] = []

        for rule in self._rules:
            d = rule.apply(f)
            if d is not None:
                decisions.append(d)
                log.debug(
                    "rule fired: %s  label=%s  conf=%.3f",
                    d.rule, d.label.name, d.confidence,
                )

        if not decisions:
            log.debug("no rules fired -- returning safe fallback NON_MM")
            return Decision(ColourID.NON_MM, 0.0, "none", 999)

        best = min(decisions, key=lambda d: (d.priority, -d.confidence))
        log.info(
            "selected: %s  label=%s  conf=%.3f  priority=%d",
            best.rule, best.label.name, best.confidence, best.priority,
        )
        return best

    @property
    def rules(self) -> list[Rule]:
        """read-only view of active rules in priority order"""
        return list(self._rules)
