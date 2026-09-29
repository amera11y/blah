# ASC Terminal

Standalone local AI terminal extracted from the uploaded source.

## Requirements

- Python 3.10+
- A machine capable of running `llama-cpp-python`
- Several GB of disk space for the GGUF model

## Run

```bash
python3 -m pip install -r requirements.txt
python3 asc_terminal.py
```

The program downloads the configured Phi-3 GGUF model on first run if it is not already present. The original source used an automatic `pip install` and automatic model download; this package keeps that behavior available through the normal requirements/install flow rather than hiding dependency installation inside the application.
