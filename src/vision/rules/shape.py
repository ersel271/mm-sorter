# src/vision/rules/shape.py
"""Stage 2 rules: shape and surface quality validation."""

from config.constants import ColourID
from src.vision import Decision, Features, Rule, register_rule

@register_rule
class LowCircularityRule(Rule):
    pass

@register_rule
class BadAspectRatioRule(Rule):
    pass

@register_rule
class LowSolidityRule(Rule):
    pass

@register_rule
class HighTextureRule(Rule):
    pass
