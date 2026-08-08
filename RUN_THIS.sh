#!/bin/bash
# Minimal one-policy Groq run (no OpenAI API calls).
# Requirements: docker image cacheforge-loop:latest built for this repo.
# Set your Groq key in the OPENAI_API_KEY env var before running (or edit below).

docker run --platform linux/amd64 --rm \
  -v "$(pwd):/workspace" \
  -e LLM_PROVIDER=groq \
  -e OPENAI_API_KEY="${OPENAI_API_KEY:-REPLACE_WITH_GROQ_KEY}" \
  -e OPENAI_BASE_URL="https://api.groq.com/openai/v1" \
  -e OPENAI_MODEL="openai/gpt-oss-120b" \
  -e OPENAI_MEM_MODEL="openai/gpt-oss-120b" \
  -e MEMORY_ENABLE=false \
  -e ITERATIONS=1 \
  -e CANDIDATES_PER_ITER=1 \
  -e SURROGATE_ENABLE=false \
  cacheforge-loop:latest \
  bash -c "pip install --quiet --no-warn-script-location -r /workspace/requirements.txt && cd /workspace/run_loop && python3 -u run_loop.py"
