# currently braindump of ideas
- fixes for:
    - color plane shift: 2x2 convolution is enough, introduces artifacts for very thin lines (recoverable with edge-adaptive resampling, or simple 2-3 layer convnet with ReLU)
    - vignetting: non-uniform gain
    - non-uniform focus: variable-size sharpening kernel (via conolution ?)
- is integrated camera resolution high enough to estimate issues above ?
- how much compute this thing has, and how programable is its image pipeline (FPGA?)
