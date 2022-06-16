# PoseGenerator
Overlays positions from a pre-generated pose JSON file to a video.

# Setup (Tested on macOS Monterey 12.3 and Ubuntu 20.04.4 LTS)

Requires Click, OpenCV, numpy, and MediaPipe.

```
$ pwd
~/Documents/PoseGenerator
$ python3 -m venv env
$ source env/bin/activate
$ pip install -e .
```
# Usage

Output file must be specified as either a MP4 or WEBM file.


```
posegenerator --help | [-v] [-b] [-u] INPUT_VIDEO POSES_JSON OUTPUT_VIDEO
```
ex.
```
posegenerator MocapTest.mp4 MocapTest_default.dmpe Mocap_overlay.mp4
Processed 242 frames. Output can be found in Mocap_overlay.mp4
```
