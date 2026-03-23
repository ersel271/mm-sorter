# src/vision/rules/colour.py
"""Stage 3 rule: compound colour classification using hue histogram and s/v gating."""

from config.constants import COLOUR_IDS, ColourID
from src.vision import Decision, Features, Rule, register_rule

@register_rule
class ColourClassificationRule(Rule):
    pass