# src/vision/rules/reject.py
"""Stage 1 rules: photometric strong exclusion."""

from config.constants import ColourID
from src.vision import Decision, Features, Rule, register_rule, Priority

@register_rule
class LowSaturationRule(Rule):
    name = "low_saturation"
    priority = Priority.S1

    def apply(self, f: Features) -> Decision | None:
        threshold = self._cfg.thresholds["sat_min"]
        if f.sat_mean < threshold:
            confidence = min((1.0 - f.sat_mean / threshold) ** 0.5, 1.0)
            return Decision(ColourID.NON_MM, confidence, self.name, self.priority)
        return None

@register_rule
class HighHighlightRule(Rule):
    name = "high_highlight"
    priority = Priority.S1

    def apply(self, f: Features) -> Decision | None:
        threshold = self._cfg.thresholds["highlight_max"]
        if f.highlight_ratio > threshold:
            # avoid division by zero if threshold is set to 0
            base = threshold if threshold > 0 else 1.0
            confidence = min(((f.highlight_ratio - threshold) / base) ** 0.5, 1.0)
            return Decision(ColourID.NON_MM, confidence, self.name, self.priority)
        return None

@register_rule
class NarrowHuePeakRule(Rule):
    name = "narrow_hue_peak"
    priority = Priority.S1

    def apply(self, f: Features) -> Decision | None:
        threshold = self._cfg.thresholds["hue_width_min"]
        if f.hue_peak_width < threshold:
            confidence = min((1.0 - f.hue_peak_width / threshold) ** 0.5, 1.0)
            return Decision(ColourID.NON_MM, confidence, self.name, self.priority)
        return None
