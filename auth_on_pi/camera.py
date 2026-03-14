"""
camera.py - Picamera2 capture module (headless, NoIR OV5647)
Keeps camera open between captures for fast burst shots.
Tune parameters at the top of this file.
"""
import cv2
import time
import numpy as np
from picamera2 import Picamera2
import config as cfg


# -- Camera parameters -- tune these for your setup --------------------------
AWB_GAINS     = cfg.AWB_GAINS
SHUTTER_SPEED = cfg.SHUTTER_SPEED
ANALOGUE_GAIN = cfg.ANALOGUE_GAIN
CONTRAST      = cfg.CONTRAST
CAPTURE_SIZE  = cfg.CAPTURE_SIZE
# ----------------------------------------------------------------------------

_camera = None


def _init_camera():
    cam = Picamera2()
    config = cam.create_still_configuration(
        main={'size': CAPTURE_SIZE, 'format': 'BGR888'},
    )
    cam.configure(config)
    cam.set_controls({
        'AwbEnable':    False,
        'ColourGains':  AWB_GAINS,
        'AeEnable':     False,
        'ExposureTime': SHUTTER_SPEED,
        'AnalogueGain': ANALOGUE_GAIN,
        'Contrast':     CONTRAST,
    })
    cam.start()
    time.sleep(1.0)  # let sensor settle
    print('[camera] Initialized')
    return cam


def get_camera():
    global _camera
    if _camera is None:
        _camera = _init_camera()
    return _camera


def capture():
    cam = get_camera()
    frame = cam.capture_array('main')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    print('[camera] Captured  shape={} mean={:.1f} max={}'.format(
        gray.shape, float(gray.mean()), int(gray.max())))
    return gray


def close():
    """Release camera. Call on shutdown."""
    global _camera
    if _camera is not None:
        _camera.stop()
        _camera.close()
        _camera = None
        print('[camera] Closed')
