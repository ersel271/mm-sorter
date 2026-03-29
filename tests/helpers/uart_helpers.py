# tests/helpers/uart_helpers.py

def sample_fields(**overrides) -> dict:
    base = {"id": 42, "class": 3, "conf": 0.91, "decision": 1, "x": 960, "y": 540}
    base.update(overrides)
    return base
