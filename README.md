# nvfp4

Solutions to Blackwell NVFP4 Kernel Hackathon (https://luma.com/9n27uem4?tk=UEBaUT).

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