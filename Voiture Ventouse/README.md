# Voiture Tampon

Python Script to detect cars and how long they stay parked, display a windows to show the result live, the gradient color formula has to be improved. The camera URL is Axis's camera feel free to adapt it to your needs.

It's a version with a result display window, and runs on an Apple Silicon M2 Mini mac (cost 700 euros) or on an old thing called an x86 architecture ;)

In terms of A.I. framework, it's Ultralytics and you can use the native Yolov8x model. All this is free for cities, but not for commercial use by third parties (GPL 3.0 license).

## Installation

Create a virtual env and install requirements

```bash
   python -m venv .venv
   source .venv/bin/activate # For Linux
   .venv/bin/Activate.ps1 # For Windows with PowerShell
   pip install -r requirements.txt
```

## Usage/Examples

Create a `conf/Camera.json` that contain your camera's informations. For more details check [Camera.json.example](./conf/Camera.json.example).

If you have a camera with multiple lenses (like a Q6010 or Q6100), you can set `Quad` to `true`.

```bash
   python parked_car_tracker.py <model_path> \n
   [--confidence <confidence>] \n
   [--device <device>] \n
   [--interval <interval>]
```
