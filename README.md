# TGI Simple

Inspired by Huggingface's text generation inference server, our "Text Generation Inference (TGI) Simple" is a small library for running local model servers. Designed for research only, and not meant for production use.

## Installation

```
git clone git@github.com:mrdrozdov/tgi_simple.git && cd tgi_simple

# Install requirements.
# pytorch
# transformers
# accelerate
```

## Quickstart

Start server.

```
python text_generation_inference.py
```

Make API calls.

```
curl http://localhost:8000 \
    -X POST \
    -d '{"prompts":["What is Deep Learning?"], "model_name": "gpt2"}' \
    -H 'Content-Type: application/json'
```
