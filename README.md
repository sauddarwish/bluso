# Bluso

Bluso is a local-first browser observatory for language models. It discovers local runtimes and GGUF models, launches your Saluki 27B model through llama.cpp, and shows real token probabilities, generation timing, context use, runtime slots, model metadata, and host resource use.

## Start on macOS or Linux

```sh
./bluso
```

Bluso opens at `http://127.0.0.1:3417`. Everything stays on loopback. It uses only Python's standard library; no package install is required.

Saluki defaults to a 4,096-token context to fit comfortably on a 16 GB Mac. Override it when needed with `BLUSO_CONTEXT=8192 ./bluso`.

To start only the dashboard without launching a model:

```sh
python3 bluso.py
```

Useful flags:

```sh
python3 bluso.py --auto-start --no-browser --port 3417
```

## Supported adapters

- **llama.cpp**: live chat stream, top-10 logprobs, slots, Prometheus metrics, timings, model properties
- **Ollama**: live chat/thinking stream, installed/running models, timings (Ollama's public API does not expose token logprobs)
- **LM Studio / vLLM / other OpenAI-compatible servers**: automatic endpoint discovery and chat streaming
- **GGUF**: direct header parsing for architecture, layers, dimensions, attention, parameter count, tensors, context and raw metadata

Bluso scans localhost ports only: llama.cpp on `8081`, `9931`, and `8080`; Ollama on `11434`; OpenAI-compatible runtimes such as LM Studio, vLLM, Jan, GPT4All, text-generation-webui, and KoboldCpp on their common local ports (`1234`, `8000`, `5000`, `5001`, `4891`, and `1337`).

## What “thinking” and “layers” mean

Bluso shows model-emitted reasoning when a backend exposes a reasoning stream. It does not claim to reveal private or hidden chain-of-thought. The anatomy page shows actual GGUF layer metadata. Standard llama.cpp and Ollama HTTP APIs do not expose per-layer hidden activations; adding those requires an instrumented inference build.

## Security

The dashboard and launched llama.cpp service bind to `127.0.0.1`. Bluso does not upload prompts, outputs, model data, or telemetry.
