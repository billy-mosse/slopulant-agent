#!/bin/sh
# Local OpenAI-compatible LLM server on :8080 (Apple Silicon).
# On the hackathon box, run vLLM instead and point LLM_BASE_URL / LLM_MODEL at it.
MODEL="${LLM_MODEL:-mlx-community/Qwen3.5-4B-MLX-4bit}"
exec mlx_lm.server --model "$MODEL" --port 8080 --chat-template-args '{"enable_thinking": false}'
