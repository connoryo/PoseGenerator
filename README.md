# PoseGenerator
Overlays positions from a pre-generated pose JSON file to a video.

# Setup (Only tested on Mac)

Requires Click, OpenCV, and numpy.

```
pip install -e .
```
# Usage

Output MUST be in MP4 format.

```
posegenerator --help | [-v] [INPUT_VIDEO] [POSES_JSON] [OUTPUT_VIDEO]
```
ex.
```
posegenerator MocapTest.mp4 MocapTest_default.dmpe Mocap_overlay.mp4
Processed 242 frames. Output can be found in Mocap_overlay.mp4
```
