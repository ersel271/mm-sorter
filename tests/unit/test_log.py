# tests/unit/test_log.py

import re
import logging
from pathlib import Path

import pytest

from utils import log as log_module
from utils.log import setup_logger

@pytest.fixture(autouse=True)
def isolate_logger():
    log_module._initialised = False
    root = logging.getLogger()
    for h in root.handlers[:]:
        h.close()
        root.removeHandler(h)
    for pkg in log_module._PROJECT_PACKAGES:
        logging.getLogger(pkg).setLevel(logging.NOTSET)
    yield
    log_module._initialised = False
    root = logging.getLogger()
    for h in root.handlers[:]:
        h.close()
        root.removeHandler(h)
    for pkg in log_module._PROJECT_PACKAGES:
        logging.getLogger(pkg).setLevel(logging.NOTSET)

@pytest.mark.smoke
class TestSetupLogger:
    """verify handler creation, log directory setup, and idempotency."""

    def test_returns_root_logger(self, tmp_cfg):
        root = setup_logger(tmp_cfg)
        assert root is logging.getLogger()

    def test_creates_log_directory(self, tmp_cfg):
        setup_logger(tmp_cfg)
        log_dir = Path(tmp_cfg.system["log_dir"])
        assert log_dir.exists()

    def test_creates_timestamped_log_file(self, tmp_cfg):
        setup_logger(tmp_cfg)
        log_dir = Path(tmp_cfg.system["log_dir"])
        log_files = list(log_dir.glob("sorter_*.log"))
        assert len(log_files) == 1
        name = log_files[0].name
        assert name.startswith("sorter_")
        assert name.endswith(".log")
        ts_part = name[len("sorter_"):-len(".log")]
        assert len(ts_part) == 15

    def test_has_file_handler(self, tmp_cfg):
        setup_logger(tmp_cfg)
        root = logging.getLogger()
        fh = [h for h in root.handlers if isinstance(h, logging.FileHandler)]
        assert len(fh) == 1

    def test_has_stream_handler(self, tmp_cfg):
        setup_logger(tmp_cfg)
        root = logging.getLogger()
        sh = [h for h in root.handlers if type(h) is logging.StreamHandler]
        assert len(sh) == 1

    def test_idempotent(self, tmp_cfg):
        setup_logger(tmp_cfg)
        setup_logger(tmp_cfg)
        root = logging.getLogger()
        our = [h for h in root.handlers if type(h) in (logging.StreamHandler, logging.FileHandler)]
        assert len(our) == 2

    def test_root_level_is_warning(self, tmp_cfg):
        setup_logger(tmp_cfg)
        assert logging.getLogger().level == logging.WARNING

    def test_project_loggers_use_requested_level(self, tmp_cfg):
        setup_logger(tmp_cfg)
        for pkg in log_module._PROJECT_PACKAGES:
            assert logging.getLogger(pkg).level == logging.DEBUG

    def test_stream_handler_is_info(self, tmp_cfg):
        setup_logger(tmp_cfg)
        root = logging.getLogger()
        sh = [h for h in root.handlers if type(h) is logging.StreamHandler]
        assert sh[0].level == logging.INFO

@pytest.mark.smoke
class TestLogOutput:
    """verify log messages reach the file with correct format."""

    def test_writes_to_file(self, tmp_cfg):
        setup_logger(tmp_cfg)
        logging.getLogger("src.test").info("hello from test")
        for h in logging.getLogger().handlers:
            h.flush()
        log_dir = Path(tmp_cfg.system["log_dir"])
        content = next(iter(log_dir.glob("sorter_*.log"))).read_text()
        assert "hello from test" in content

    def test_format_contains_file_and_line(self, tmp_cfg):
        setup_logger(tmp_cfg)
        logging.getLogger("mymod").warning("check format")

        for h in logging.getLogger().handlers:
            h.flush()

        log_dir = Path(tmp_cfg.system["log_dir"])
        content = next(iter(log_dir.glob("sorter_*.log"))).read_text()

        pattern = rf"{re.escape(Path(__file__).name)}\s*:\s*\d+"
        assert re.search(pattern, content)
        assert "WARNING" in content

    def test_debug_reaches_file(self, tmp_cfg):
        setup_logger(tmp_cfg)
        logging.getLogger("utils.dbg").debug("debug msg")
        for h in logging.getLogger().handlers:
            h.flush()
        log_dir = Path(tmp_cfg.system["log_dir"])
        content = next(iter(log_dir.glob("sorter_*.log"))).read_text()
        assert "debug msg" in content

    def test_external_logger_below_warning_not_written(self, tmp_cfg):
        setup_logger(tmp_cfg)
        logging.getLogger("somelib").info("external info")
        for h in logging.getLogger().handlers:
            h.flush()
        log_dir = Path(tmp_cfg.system["log_dir"])
        content = next(iter(log_dir.glob("sorter_*.log"))).read_text()
        assert "external info" not in content
