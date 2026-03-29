# src/vision/rules/shape.py
"""Stage 2 rules: shape and surface quality validation."""

from config.constants import ColourID
from src.vision import Decision, Features, Rule, register_rule

@register_rule
class LowCircularityRule(Rule):
    name = "low_circularity"
    priority = 20

    def apply(self, f: Features) -> Decision | None:
        threshold = self._cfg.thresholds["circularity_min"]
        if f.circularity < threshold:
            confidence = min((1.0 - f.circularity / threshold) ** 0.5, 1.0)
            return Decision(ColourID.NON_MM, confidence, self.name, self.priority)
        return None

@register_rule
class BadAspectRatioRule(Rule):
    name = "bad_aspect_ratio"
    priority = 20

    def apply(self, f: Features) -> Decision | None:
        threshold = self._cfg.thresholds["aspect_ratio_max"]
        if f.aspect_ratio > threshold:
            confidence = min(((f.aspect_ratio - threshold) / threshold) ** 0.5, 1.0)
            return Decision(ColourID.NON_MM, confidence, self.name, self.priority)
        return None

@register_rule
class LowSolidityRule(Rule):
    name = "low_solidity"
    priority = 20

    def apply(self, f: Features) -> Decision | None:
        threshold = self._cfg.thresholds["solidity_min"]
        if f.solidity < threshold:
            confidence = min((1.0 - f.solidity / threshold) ** 0.5, 1.0)
            return Decision(ColourID.NON_MM, confidence, self.name, self.priority)
        return None

@register_rule
class HighTextureRule(Rule):
    name = "high_texture"
    priority = 20

    def apply(self, f: Features) -> Decision | None:
        threshold = self._cfg.thresholds["texture_max"]
        if f.texture_variance > threshold:
            base = threshold if threshold > 0 else 1.0
            confidence = min(((f.texture_variance - threshold) / base) ** 0.5, 1.0)
            return Decision(ColourID.NON_MM, confidence, self.name, self.priority)
        return None
