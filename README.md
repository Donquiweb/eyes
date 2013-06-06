# eyes

**eyes** is an OpenCV project

It currently features the following programs:

- **objectTracker**: Ability to track objects, provided you use the filter
  settings to filter out the specific object's colour out.

- **stereoVision**: Using two webcams, it currently is only able to display two
  webcam feeds both at the same time.

- **cameraCalibration**: Calibrate camera to obtain intrinsic and distortion
  settings. The results are saved into "calibration.xml" and "distortion.xml"
  respecitively.

## Requirements

- `cmake`: version 2.6 and higher
- [`dbg`](https://github.com/chutsu/dbg): debug utilities


## Build and Install
#### Generate make files using `cmake`
First navigate into
`al` root, and generate make files using `cmake` **Note: the dot '.' is
important**!

    cd <path to al> cmake .

where `<path to al>` is the path to `al`.


## LICENCE
MIT LICENCE Copyright (C) <2012> Chris Choi

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

