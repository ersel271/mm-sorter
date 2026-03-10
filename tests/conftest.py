# tests/conftest.py
"""
Pytest configuration entry point.
Loads shared fixture modules used across the test suite.
"""

pytest_plugins = [
    "tests.fixtures.config_fixtures",
    "tests.fixtures.uart_fixtures",
    "tests.fixtures.vision_fixtures",
]
