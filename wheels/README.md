# Offline PaddlePaddle GPU wheel

`requirements.linux.cuda12.txt` installs the CUDA-12 PaddlePaddle GPU build
(`paddlepaddle-gpu==2.6.2.post120`) from Paddle's own package index, which is
China-hosted (`paddlepaddle.org.cn` / `bcebos.com`). Some networks — and many
datacenter / VM environments — cannot reach it, so that install silently gets no
CUDA build and ALPR falls back to CPU.

**If that's your situation:** download the wheel matching your module venv's Python
from a machine that *can* reach the host, and drop it in **this folder**. `install.sh`
installs it automatically when the indexed download isn't available.

Wheel (Python 3.8 shown — pick the `cp##` tag matching your venv):

```
paddlepaddle_gpu-2.6.2.post120-cp38-cp38-linux_x86_64.whl
```

Direct URL (when reachable):

```
https://paddle-whl.bj.bcebos.com/stable/cu120/paddlepaddle-gpu/paddlepaddle_gpu-2.6.2.post120-cp38-cp38-linux_x86_64.whl
```

Only the `paddlepaddle-gpu` wheel is needed here; everything else installs from PyPI.
Do **not** switch to PaddlePaddle 3.x — it drops Python 3.8 and breaks PaddleOCR 2.7.0.3.
