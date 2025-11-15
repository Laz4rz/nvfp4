# nvfp4

Solutions to Blackwell NVFP4 Kernel Hackathon (https://luma.com/9n27uem4?tk=UEBaUT). Base code from https://github.com/gpu-mode/reference-kernels/tree/main/problems/nvidia.

Test locally (blackwell GPU) with:
```bash
PYTHONPATH=".:nvfp4_gemv" \
POPCORN_FD=1 \
python eval.py test minimal.txt
```

Benchmark locally with:
```bash
PYTHONPATH=".:nvfp4_gemv" \
POPCORN_FD=1 \
python eval.py benchmark minimal.txt
```

Interact with python-side code by running debug on `debug.py`.