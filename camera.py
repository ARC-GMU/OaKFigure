"""Frame sources: OAK-D Pro (DepthAI v3), with webcam / video-file fallback.

Every source runs a background thread and only keeps the newest frames, so
nothing downstream ever works on stale video.

From the OAK we stream:
  rgb    1280x720 colour (centre camera)
  depth  uint16 millimetres, aligned to the rgb image (0 = unknown)
  left / right   640x400 grayscale from the two stereo cameras, which also see
                 infrared, so they show the Pro's IR dot projector / flood light
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Frames:
    rgb: np.ndarray  # HxWx3 BGR
    depth: np.ndarray | None = None  # HxW uint16 mm, same size as rgb
    left: np.ndarray | None = None  # grayscale
    right: np.ndarray | None = None  # grayscale
    intrinsics: tuple[float, float, float, float] | None = None  # fx, fy, cx, cy for rgb/depth

    @property
    def has_depth(self) -> bool:
        return self.depth is not None


class FrameSource:
    """Base class: subclasses implement _open(), _read() and _close()."""

    has_depth = False
    has_ir = False

    def __init__(self) -> None:
        self._frames: Frames | None = None
        self._frame_id = 0
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None
        self.error: Exception | None = None

    # -- subclass hooks -----------------------------------------------------
    def _open(self) -> None: ...
    def _read(self) -> Frames | None: ...
    def _close(self) -> None: ...

    def set_ir(self, dots: float, flood: float) -> None:
        """IR dot projector / flood light intensity, 0..1 (OAK-D Pro only)."""

    # -- public API ---------------------------------------------------------
    def start(self) -> "FrameSource":
        self._open()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def _loop(self) -> None:
        try:
            while self._running:
                frames = self._read()
                if frames is None:
                    time.sleep(0.002)
                    continue
                with self._lock:
                    self._frames = frames
                    self._frame_id += 1
        except Exception as exc:  # surfaced to the UI instead of dying silently
            self.error = exc
            self._running = False

    def latest(self) -> tuple[int, Frames | None]:
        """Return (frame_id, frames). frame_id increases with each new bundle."""
        with self._lock:
            return self._frame_id, self._frames

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2)
        self._close()


class OakSource(FrameSource):
    has_depth = True
    has_ir = True

    def __init__(self, rgb_size=(1280, 720), mono_size=(640, 400), fps: int = 30,
                 usb2: bool | None = None) -> None:
        """usb2: None = try USB3 first and fall back to USB2; True = USB2 only."""
        super().__init__()
        self.rgb_size = rgb_size
        self.mono_size = mono_size
        self.fps = fps
        self.usb2 = usb2
        self._pipeline = None
        self._device = None
        self._queues: dict = {}
        self._latest: dict[str, np.ndarray] = {}
        self._intrinsics = None
        self._ir_state = None

    def _open(self) -> None:
        import depthai as dai

        if not hasattr(dai.node.Camera, "build"):
            raise RuntimeError("DepthAI v3 is required: pip install -U 'depthai>=3'")

        if not self.usb2:
            try:
                return self._build(dai, usb2=False)
            except Exception as exc:
                print(f"[camera] USB3 connection failed ({exc}).\n"
                      "[camera] Falling back to USB2 with smaller video. For full quality use a USB3 port\n"
                      "[camera] (usually blue) and a USB3-rated USB-C cable, or a powered USB3 hub.")
                self._close()
                time.sleep(2)  # give the camera time to reboot
        self._build(dai, usb2=True)

    def _build(self, dai, usb2: bool) -> None:
        rgb_type = dai.ImgFrame.Type.BGR888p
        if usb2:  # USB2 carries ~35 MB/s, so shrink and compress the streams
            self.rgb_size, self.fps, rgb_type = (640, 360), 20, dai.ImgFrame.Type.NV12
        self._device = dai.Device(dai.UsbSpeed.HIGH) if usb2 else dai.Device()
        p = self._pipeline = dai.Pipeline(self._device)
        rgb_cam = p.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
        left_cam = p.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
        right_cam = p.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)

        try:  # undistorted colour lines up better with the depth image
            rgb_out = rgb_cam.requestOutput(self.rgb_size, rgb_type, fps=self.fps, enableUndistortion=True)
        except TypeError:
            rgb_out = rgb_cam.requestOutput(self.rgb_size, rgb_type, fps=self.fps)
        left_out = left_cam.requestOutput(self.mono_size, fps=self.fps)
        right_out = right_cam.requestOutput(self.mono_size, fps=self.fps)

        stereo = p.create(dai.node.StereoDepth)
        for preset in ("DEFAULT", "HIGH_DETAIL", "HIGH_DENSITY"):
            if hasattr(dai.node.StereoDepth.PresetMode, preset):
                stereo.setDefaultProfilePreset(getattr(dai.node.StereoDepth.PresetMode, preset))
                break
        stereo.setLeftRightCheck(True)
        left_out.link(stereo.left)
        right_out.link(stereo.right)
        # line the depth image up with the colour image
        if hasattr(stereo, "inputAlignTo"):
            rgb_out.link(stereo.inputAlignTo)
        else:
            stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)

        q = lambda out: out.createOutputQueue(maxSize=2, blocking=False)  # noqa: E731
        self._queues = {"rgb": q(rgb_out), "depth": q(stereo.depth), "left": q(left_out), "right": q(right_out)}
        p.start()
        print(f"[camera] {self._device.getDeviceName()} connected at {self._device.getUsbSpeed().name}, "
              f"colour {self.rgb_size[0]}x{self.rgb_size[1]} @ {self.fps} fps")
        try:
            calib = self._device.readCalibration()
            m = calib.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A, *self.rgb_size)
            self._intrinsics = (m[0][0], m[1][1], m[0][2], m[1][2])
        except Exception as exc:
            print(f"[camera] could not read calibration ({exc}); using an estimate")
            w, h = self.rgb_size
            f = w / (2 * np.tan(np.radians(69 / 2)))  # ~69 deg horizontal FOV
            self._intrinsics = (f, f, w / 2, h / 2)
        self.set_ir(0.6, 0.0)

    def set_ir(self, dots: float, flood: float) -> None:
        if self._device is None or self._ir_state == (dots, flood):
            return
        self._ir_state = (dots, flood)
        d = self._device
        try:
            if hasattr(d, "setIrLaserDotProjectorIntensity"):
                d.setIrLaserDotProjectorIntensity(dots)
                d.setIrFloodLightIntensity(flood)
            else:  # older API, in mA
                d.setIrLaserDotProjectorBrightness(dots * 1200)
                d.setIrFloodLightBrightness(flood * 1500)
        except Exception as exc:  # e.g. an OAK-D without the "Pro" IR hardware
            print(f"[camera] IR control unavailable: {exc}")
            self.has_ir = False

    def _read(self) -> Frames | None:
        for name, queue in self._queues.items():
            msg = queue.tryGet()
            if msg is None:
                continue
            if name == "depth":
                img = msg.getFrame()
            else:
                img = msg.getCvFrame()
                if name != "rgb" and img.ndim == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            self._latest[name] = img
            if name == "rgb":
                self._latest["rgb_new"] = True

        if not self._latest.pop("rgb_new", False):
            return None
        rgb = self._latest["rgb"]
        depth = self._latest.get("depth")
        if depth is not None and depth.shape[:2] != rgb.shape[:2]:
            depth = cv2.resize(depth, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
            self._latest["depth"] = depth
        return Frames(rgb, depth, self._latest.get("left"), self._latest.get("right"), self._intrinsics)

    def _close(self) -> None:
        if self._device is not None:
            try:
                self.set_ir(0.0, 0.0)
            except Exception:
                pass
        for obj, method in ((self._pipeline, "stop"), (self._device, "close")):
            if obj is not None:
                try:
                    getattr(obj, method)()
                except Exception:
                    pass
        self._pipeline = self._device = None
        self._ir_state = None


class OpenCVSource(FrameSource):
    """Webcam index or video file, for testing without the OAK (no depth / IR views)."""

    def __init__(self, source: int | str, loop_video: bool = True) -> None:
        super().__init__()
        self.source = source
        self.loop_video = loop_video
        self._cap: cv2.VideoCapture | None = None
        self._is_file = isinstance(source, str)
        self._frame_period = 0.0

    def _open(self) -> None:
        self._cap = cv2.VideoCapture(self.source)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open video source {self.source!r}")
        if self._is_file:
            self._frame_period = 1.0 / (self._cap.get(cv2.CAP_PROP_FPS) or 30)

    def _read(self) -> Frames | None:
        ok, frame = self._cap.read()
        if not ok:
            if self._is_file and self.loop_video:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return None
        if self._is_file:  # play files at real speed
            time.sleep(self._frame_period)
        return Frames(frame)

    def _close(self) -> None:
        if self._cap is not None:
            self._cap.release()


def make_source(spec: str, usb2: bool = False) -> FrameSource:
    """'oak' -> OAK-D Pro, a number -> webcam index, anything else -> video file path."""
    if spec == "oak":
        return OakSource(usb2=True if usb2 else None)
    if spec.isdigit():
        return OpenCVSource(int(spec))
    return OpenCVSource(spec)
