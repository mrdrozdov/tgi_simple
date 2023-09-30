# TGI Simple

Inspired by Huggingface's text generation inference server, our "Text Generation Inference (TGI) Simple" is a small library for running local model servers. Designed for research only, and not meant for production use.

## Installation

```
git clone git@github.com:mrdrozdov/tgi_simple.git && cd tgi_simple

# Install requirements.
# fire
# pytorch
# transformers
# accelerate
```

## Quickstart

Start server.

```
# Sampling.
python text_generation_inference.py --model_type gpt2 --model_name_or_path gpt2 --use-cpu

# Greedy generation.
python text_generation_inference.py --model_type gpt2 --model_name_or_path gpt2 --do_greedy --max_new_tokens 128
```

Make API calls.

```
curl http://localhost:8000 \
    -X POST \
    -d '{"prompts":["What is Deep Learning?"], "model_name": "gpt2"}' \
    -H 'Content-Type: application/json'
```
