# src/vision/rules/reject.py
"""Stage 1 rules: photometric strong exclusion."""

from config.constants import ColourID
from src.vision import Decision, Features, Rule, register_rule

@register_rule
class LowSaturationRule(Rule):
    pass

@register_rule
class HighHighlightRule(Rule):
    pass

@register_rule
class NarrowHuePeakRule(Rule):
    pass
