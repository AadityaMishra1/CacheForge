# src/ollama_cluster.py
from __future__ import annotations
import os
import random
import sys
from typing import List, Tuple, Optional

import ollama
import requests

# Defaults; you can override via environment or from common.py when calling.
HOSTS = [
    "gpu16","gpu17","gpu18","gpu19","gpu20","gpu21","gpu22","gpu23",
    "gpu24","gpu25","gpu26","gpu27","gpu28","gpu29","gpu30","gpu31",
    "gpu32","gpu33","gpu34","gpu35","gpu36"
]
PORTS = [11432, 11434, 11431, 11433]

_active_server: Optional[Tuple[str, int]] = None

def _ping_base_url(url: str, timeout: float = 1.5) -> bool:
    """Return True if Ollama server responds to /api/version quickly."""
    try:
        r = requests.get(f"{url}/api/version", timeout=timeout)
        return r.ok
    except Exception as e:
        print(f"[ping error] {url}: {e}")
        return False

def _try_model_minimal(host: str, port: int, model: str) -> bool:
    """
    Try a minimal generate call with num_predict=1. This also surfaces 'model not found'
    errors early so we can move to the next server that has the model.
    """
    try:
        client = ollama.Client(host=f"http://{host}:{port}")
        _ = client.generate(model=model, prompt="ping", options={"num_predict": 1})
        return True
    except Exception as e:
        print(f"[ping error] {url}: {e}")
        return False

def get_client_for_model(
    model: str,
    hosts: Optional[List[str]] = None,
    ports: Optional[List[int]] = None,
    force_refresh: bool = False
) -> Tuple[ollama.Client, str]:
    """
    Returns (client, base_url) for a server that responds and can run `model`.
    Caches the last good server; if it fails or force_refresh=True, it searches again.
    """
    global _active_server
    hosts = list(hosts or HOSTS)
    ports = list(ports or PORTS)

    # Shuffle to spread load
    rng = random.Random(os.urandom(8))
    rng.shuffle(hosts)
    rng.shuffle(ports)

    # Try cached server first
    if _active_server and not force_refresh:
        host, port = _active_server
        base = f"http://{host}:{port}"
        if _ping_base_url(base) and _try_model_minimal(host, port, model):
            return ollama.Client(host=base), base
        # drop cache and probe again
        _active_server = None

    # Probe all host×port combos
    for h in hosts:
        for p in ports:
            base = f"http://{h}:{p}"
            if not _ping_base_url(base):
                continue
            if not _try_model_minimal(h, p, model):
                continue
            _active_server = (h, p)
            return ollama.Client(host=base), base

    raise RuntimeError("No available Ollama servers found in HPC list that can run the requested model.")

def invalidate_cache():
    global _active_server
    _active_server = None