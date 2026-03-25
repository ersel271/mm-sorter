# src/vision/rule.py
"""
Base class, registry, and Decision type for classifier rules.

Each rule receives a Features object, evaluates a single condition or a
compound check, and returns a Decision. Rules are registered via the
@register_rule decorator and instantiated by Classifier at construction time.

Usage:
    @register_rule
    class MyRule(Rule):
        name = "my_rule"
        priority = 10
        def apply(self, f: Features) -> Decision | None: ...
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

from config import Config
from .features import Features
from config.constants import ColourID

log = logging.getLogger(__name__)

_RULE_REGISTRY: list[type["Rule"]] = []

@dataclass(frozen=True)
class Decision:
    """
    immutable result produced by a single rule evaluation.

    label=ColourID.NON_MM means the rule fired and the object should be
    rejected. a rule that does not fire returns None instead of a Decision.
    """

    label: ColourID
    confidence: float
    rule: str
    priority: int

class Rule(ABC):
    """
    abstract base for all classifier rules.
    """

    name: ClassVar[str]
    priority: ClassVar[int]

    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg

    @abstractmethod
    def apply(self, f: Features) -> Decision | None:
        """
        evaluate this rule against the given features.

        must never raise (unless a fatal, unrecoverable error occurs).
        returns None when the rule does not fire.
        deterministic: same Features always produces the same Decision.
        """
        ...

def register_rule(cls: type[Rule]) -> type[Rule]:
    """
    class decorator that registers a Rule subclass in the global registry.
    to disable a rule, remove this decorator
    """
    _RULE_REGISTRY.append(cls)
    return cls
