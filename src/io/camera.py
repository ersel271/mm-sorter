# src/io/camera.py
"""
Camera capture module for the vision pipeline.

Wraps cv2.VideoCapture with config-driven setup, property control,
and structured error handling. The pipeline calls open() once at
startup and read() in the main loop.

Usage:
    cam = Camera(cfg)
    cam.open()
    ok, frame = cam.read()
    cam.release()
"""

import logging

import cv2
import numpy as np

from config import Config

log = logging.getLogger(__name__)

class Camera:
    """
    USB camera interface. configures resolution, format, focus,
    exposure, and white balance from the config file. all setter
    methods return True on success, False on failure and never raise.
    """

    def __init__(self, config: Config):
        self._cfg = config.camera
        self._cap: cv2.VideoCapture | None = None
        
        log.info("camera initialised -- device=%d", self._cfg["device"])

    def open(self) -> bool:
        """
        open the camera device and apply all settings from config.
        returns True if the camera is ready to capture.
        """
        device = int(self._cfg["device"])
        self._cap = cv2.VideoCapture(device)  # type: ignore[call-arg]

        if not self._cap.isOpened():
            log.error("failed to open camera device %d", device)
            self._cap = None
            return False

        self._apply_format()
        self._apply_focus()
        self._apply_exposure()
        self._apply_white_balance()
        self._apply_power_line_freq()

        props = self.get_properties()
        log.info(
            "camera opened -- %dx%d @ %d fps, format=%s, focus=%d",
            props["width"], props["height"], props["fps"],
            props["format"], props["focus"],
        )
        return True

    def read(self) -> tuple[bool, np.ndarray | None]:
        """
        grab a single frame. returns (True, BGR frame) on success
        or (False, None) on failure.
        """
        if self._cap is None:
            return False, None

        ok, frame = self._cap.read()
        if not ok:
            log.warning("frame read failed")
            return False, None

        return True, frame

    def set_focus(self, value: int) -> bool:
        """
        set manual focus to the given value (1--1023).
        disables autofocus first.
        """
        if self._cap is None:
            return False

        self._cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        ok = bool(self._cap.set(cv2.CAP_PROP_FOCUS, value))
        if ok:
            log.info("focus set to %d", value)
        else:
            log.warning("failed to set focus to %d", value)
        return ok

    def set_autofocus(self, enabled: bool) -> bool:
        """
        enable or disable autofocus.
        """
        if self._cap is None:
            return False

        val = 1 if enabled else 0
        ok = bool(self._cap.set(cv2.CAP_PROP_AUTOFOCUS, val))
        log.info("autofocus %s", "enabled" if enabled else "disabled")
        return ok

    def set_exposure(self, mode: int, value: int | None = None) -> bool:
        """
        set exposure mode. mode 3 = aperture priority (auto),
        mode 1 = manual. when manual, value sets the exposure time.
        """
        if self._cap is None:
            return False

        ok = bool(self._cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, mode))
        if mode == 1 and value is not None:
            self._cap.set(cv2.CAP_PROP_EXPOSURE, value)
        log.info("exposure mode=%d value=%s", mode, value)
        return ok

    def set_white_balance(self, auto: bool, temperature: int | None = None) -> bool:
        """
        configure white balance. when auto is False, temperature
        is applied as a fixed value.
        """
        if self._cap is None:
            return False

        if auto:
            self._cap.set(cv2.CAP_PROP_AUTO_WB, 1)
            log.info("white balance set to auto")
        else:
            self._cap.set(cv2.CAP_PROP_AUTO_WB, 0)
            if temperature is not None:
                self._cap.set(cv2.CAP_PROP_WB_TEMPERATURE, temperature)
            log.info("white balance set to %dK", temperature or 0)
        return True

    def get_properties(self) -> dict:
        """
        read current camera properties from the driver.
        """
        if self._cap is None:
            return {}

        fourcc_int = int(self._cap.get(cv2.CAP_PROP_FOURCC))
        fourcc_str = "".join(chr((fourcc_int >> (8 * i)) & 0xFF) for i in range(4))

        return {
            "width": int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": int(self._cap.get(cv2.CAP_PROP_FPS)),
            "format": fourcc_str,
            "focus": int(self._cap.get(cv2.CAP_PROP_FOCUS)),
            "autofocus": int(self._cap.get(cv2.CAP_PROP_AUTOFOCUS)),
            "auto_exposure": int(self._cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)),
            "auto_wb": int(self._cap.get(cv2.CAP_PROP_AUTO_WB)),
        }

    def release(self) -> None:
        """
        release the camera device.
        """
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            log.info("camera released")

    @property
    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def _apply_format(self) -> None:
        if self._cap is None:
            raise RuntimeError("camera is not open")
        fourcc = cv2.VideoWriter_fourcc(*self._cfg["format"])
        self._cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._cfg["width"])
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._cfg["height"])
        self._cap.set(cv2.CAP_PROP_FPS, self._cfg["fps"])

    def _apply_focus(self) -> None:
        if self._cap is None:
            raise RuntimeError("camera is not open")
        if self._cfg["autofocus"]:
            self._cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        else:
            self._cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
            self._cap.set(cv2.CAP_PROP_FOCUS, self._cfg["focus"])

    def _apply_exposure(self) -> None:
        if self._cap is None:
            raise RuntimeError("camera is not open")
        mode = self._cfg.get("auto_exposure", 3)
        self._cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, mode)
        if mode == 1:
            exposure = self._cfg.get("exposure", 157)
            self._cap.set(cv2.CAP_PROP_EXPOSURE, exposure)

    def _apply_white_balance(self) -> None:
        if self._cap is None:
            raise RuntimeError("camera is not open")
        if self._cfg.get("auto_wb", True):
            self._cap.set(cv2.CAP_PROP_AUTO_WB, 1)
        else:
            self._cap.set(cv2.CAP_PROP_AUTO_WB, 0)
            temp = self._cfg.get("wb_temperature", 4600)
            self._cap.set(cv2.CAP_PROP_WB_TEMPERATURE, temp)

    def _apply_power_line_freq(self) -> None:
        # TODO: CAP_PROP_* has no power line frequency option so we need platform-specific handling
        pass
