# src/vision/rules/colour.py
"""Stage 3 rule: compound colour classification using hue histogram and s/v gating."""

from config.constants import COLOUR_IDS, ColourID
from src.vision import Decision, Features, Rule, register_rule

@register_rule
class ColourClassificationRule(Rule):
    name = "colour_match"
    priority = 30

    def apply(self, f: Features) -> Decision:
        min_conf = self._cfg.thresholds["colour_confidence_min"]
        epsilon = self._cfg.thresholds["colour_ambiguity_epsilon"]

        best_label: ColourID | None = None
        best_conf = 0.0
        second_conf = 0.0

        for colour_name, colour_cfg in self._cfg.colours.items():
            s_min, s_max = colour_cfg["s"]
            v_min, v_max = colour_cfg["v"]

            # gate: skip colour if sat or val is outside this colour's expected range
            if not (s_min <= f.sat_mean <= s_max):
                continue
            if not (v_min <= f.val_mean <= v_max):
                continue

            # score: fraction of hue histogram mass in this colour's hue ranges
            overlap = 0.0
            for h_min, h_max in colour_cfg["h"]:
                overlap += float(f.hue_hist[h_min : h_max + 1].sum())
            conf = min(overlap, 1.0)

            if conf > best_conf:
                second_conf = best_conf
                best_conf = conf
                best_label = ColourID(COLOUR_IDS[colour_name.lower()])
            elif conf > second_conf:
                second_conf = conf

        if best_label is None or best_conf < min_conf:
            return Decision(None, 0.0, self.name, self.priority)

        # ambiguity: margin between top two candidates is too small to trust
        if best_conf - second_conf < epsilon:
            return Decision(ColourID.NON_MM, 0.0, "ambiguous_colour", self.priority)

        return Decision(best_label, best_conf, self.name, self.priority)
