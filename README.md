# OaKFigure

A STEM-event demo for the OAK-D Pro camera. Six views, each with a title and a one-line explanation for kids:

| # | View | What it shows |
|---|---|---|
| 1 | **Stick Figures** | Everyone as a pastel stick figure. Each person keeps their colour, even after leaving and coming back. |
| 2 | **Skeleton Vision** | The real camera picture with stick-figure skeletons drawn on top |
| 3 | **Depth Rainbow** | The depth map in rainbow colours (red = close, blue = far) with a distance readout in the middle. Up/Down change the range. |
| 4 | **How Far Away?** | Stick figures with a label showing how far away each person is |
| 5 | **3D Dots** | A rotating 3D point cloud of the room. Drag with the mouse to spin it; **T** switches to rainbow colours. |
| 6 | **Two Eyes** | The left and right stereo cameras side by side. **W** flips between them so you can see the shift. |

**Mouse:** click a tab at the bottom to change view, and **Exit** (top right) to quit.
**Keys:** 1–6 or ←/→ (or Space) change view · **A** auto-cycle · **M** mirror · **F** fullscreen · **H** hide the bars · **Q/Esc** quit

## How it works

| File | Role |
|---|---|
| `camera.py` | One DepthAI v3 pipeline: 1280×720 colour, stereo depth lined up with the colour image, and both 640×400 mono cameras. The IR dot projector stays on to improve depth. A webcam or video file also works for testing; the depth and stereo views are then greyed out. |
| `pose.py` | Runs YOLO11 pose with ByteTrack on a background thread, smooths the joints, and measures each person's distance from the depth map |
| `identity.py` | Permanent identities and colours, matched by clothing colour so returning people keep their colour |
| `figures.py` | Anti-aliased stick-figure drawing and fitting the camera image to the screen |
| `views.py` | All the views. To add one, subclass `View` and add it to `make_views()`. |
| `main.py` | Window, flat title bar with Exit button, bottom tab bar, keyboard/mouse input, auto-cycle |

## Setup

```bash
# one-time: let your user access the OAK over USB
echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"' | sudo tee /etc/udev/rules.d/80-movidius.rules
sudo udevadm control --reload-rules && sudo udevadm trigger

# Python environment (needs python3-venv, or use uv)
sudo apt install python3.14-venv          # or: curl -LsSf https://astral.sh/uv/install.sh | sh
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
#   uv alternative: uv venv && uv pip install -r requirements.txt
```

The first run downloads `yolo11n-pose.pt` (~6 MB).

## Run

```bash
.venv/bin/python main.py                 # OAK-D Pro (use a USB3 port)
.venv/bin/python main.py --source 0      # webcam, for testing without the OAK
.venv/bin/python main.py --fullscreen
.venv/bin/python main.py --imgsz 480     # faster on CPU
.venv/bin/python main.py --model yolo11s-pose.pt   # more accurate, slower
```

Or use the launcher, which takes the same options: `./run.sh --fullscreen --auto 25`

**USB3 vs USB2:** the app tries USB3 first. If that fails, it falls back to USB2 with smaller video (640×360 colour at 20 fps). Common causes are a USB2-only USB-C cable, a USB2 port, or too little power. With a USB3 cable and port you get 1280×720 at 30 fps. On a known USB2 setup, add `--usb2` to skip the ~10 s USB3 attempt.

## At the event

```bash
.venv/bin/python main.py --fullscreen --auto 25   # kiosk: change view every 25 s, press A to stop
.venv/bin/python main.py --units m                # metres instead of feet
.venv/bin/python main.py --view 3                 # start on Depth Rainbow
```

Tips:
- Stand 1–4 m (3–13 ft) from the camera. Depth stops working closer than about 35 cm.
- The IR dot projector (invisible, Class 1 eye-safe laser) helps depth on plain walls and floors.

## Notes

- **GPU:** the code uses the GPU only if your PyTorch build supports that card. Recent PyTorch builds may no longer support Pascal cards like the GTX 1050, and then it runs on the CPU. On CPU, `yolo11n-pose` at `--imgsz 480` should handle real-time on a 16-thread machine. To try the 1050, install a CUDA 12.6 PyTorch build:
  `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126`
- **Re-identification** compares clothing colours. It works well when people wear different clothes. Two people dressed alike may swap colours after leaving and coming back. To make matching stricter, lower `match_threshold` in `IdentityManager`; to make it looser, raise it.
- Identities last only for the current session. Restarting the app gives out colours again from the start (blue first).
