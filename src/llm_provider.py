# src/llm_provider.py
from __future__ import annotations
import os
from typing import List, Dict, Optional, Tuple

class LLMProvider:
    """
    Unified interface over OpenAI and Ollama (local or HPC cluster).
    - chat(messages, ...) -> str
    - embed(text, ...) -> List[float] | None
    """
    def __init__(
        self,
        provider: str = "openai",
        openai_api_key: Optional[str] = None,
        openai_base_url: Optional[str] = None,
        host: Optional[str] = None,           # Ollama base URL for local, e.g., http://localhost:11434
        mode: str = "local",                  # "local" or "hpc" (only for ollama)
        hpc_hosts: Optional[List[str]] = None,
        hpc_ports: Optional[List[int]] = None,
        default_model: Optional[str] = None,
        default_mem_model: Optional[str] = None,
        default_embed_model: Optional[str] = None
    ):
        self.provider = (provider or "openai").lower()
        self.default_model = default_model
        self.default_mem_model = default_mem_model or default_model
        self.default_embed_model = default_embed_model
        self.mode = (mode or "local").lower()
        self._ollama_host = host  # for local
        self._ollama_client = None
        self._cluster = None      # populated when using HPC
        self._groq_client = None

        if self.provider == "openai":
            from openai import OpenAI
            api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY is not set")
            client_kwargs = {"api_key": api_key}
            # Allow OpenAI-compatible hosts (OpenRouter, Together, etc.)
            if openai_base_url or os.getenv("OPENAI_BASE_URL"):
                client_kwargs["base_url"] = openai_base_url or os.getenv("OPENAI_BASE_URL")
            self._client = OpenAI(**client_kwargs)
            self.default_embed_model = self.default_embed_model or os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

        elif self.provider == "groq":
            try:
                from groq import Groq
            except Exception as e:
                raise ImportError("groq package not installed. pip install groq") from e
            # Prefer explicit arg, then env GROQ_API_KEY, then OPENAI_API_KEY for compatibility
            api_key = openai_api_key or os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("GROQ_API_KEY (or OPENAI_API_KEY) is not set")
            self._groq_client = Groq(api_key=api_key)
            # Groq does not expose embeddings today; keep embed model None
            self.default_embed_model = None

        elif self.provider == "ollama":
            import ollama
            self._ollama = ollama
            # Local client bound to a specific host if provided
            if self.mode == "local" and self._ollama_host:
                try:
                    self._ollama_client = ollama.Client(host=self._ollama_host)
                except Exception:
                    # fall back to module-level calls; rely on OLLAMA_HOST env
                    os.environ.setdefault("OLLAMA_HOST", self._ollama_host)

            # HPC discovery handler
            if self.mode == "hpc":
                try:
                    # Works when ../src is on sys.path (as in your run_loop.py)
                    import ollama_cluster as cluster
                except ImportError:
                    try:
                        # Works when src is a package and you run with -m
                        from . import ollama_cluster as cluster
                    except Exception as e:
                        raise ImportError(
                            "Could not import ollama_cluster. Ensure '../src' is on sys.path "
                            "or add __init__.py to 'src' and run as a package."
                        ) from e

                self._cluster = {
                    "get": lambda model, force=False: cluster.get_client_for_model(
                        model, hosts=hpc_hosts, ports=hpc_ports, force_refresh=force
                    ),
                    "invalidate": cluster.invalidate_cache,
                }

            self.default_embed_model = self.default_embed_model or os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def _ollama_chat_once(self, messages: List[Dict[str, str]], model: str, temperature: float, num_ctx: Optional[int], max_tokens: Optional[int]) -> str:
        opts = {"temperature": temperature}
        if num_ctx:
            opts["num_ctx"] = num_ctx
        if max_tokens:
            opts["num_predict"] = max_tokens

        if self.mode == "hpc" and self._cluster:
            client, _ = self._cluster["get"](model, force=False)
            resp = client.chat(model=model, messages=messages, options=opts)
            return resp["message"]["content"]

        if self._ollama_client:
            resp = self._ollama_client.chat(model=model, messages=messages, options=opts)
            return resp["message"]["content"]
        # module-level call (uses OLLAMA_HOST env)
        resp = self._ollama.chat(model=model, messages=messages, options=opts)
        return resp["message"]["content"]

    def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 1,
        num_ctx: Optional[int] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        model = model or self.default_model
        if self.provider == "openai":
            kwargs = {
                "model": model,
                "messages": messages
            }
            resp = self._client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content
        if self.provider == "groq":
            # Groq client mirrors OpenAI chat API; map max_tokens -> max_completion_tokens
            kwargs = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
            }
            if max_tokens is not None:
                kwargs["max_completion_tokens"] = max_tokens
            # num_ctx not exposed in Groq API; rely on model defaults
            resp = self._groq_client.chat.completions.create(**kwargs)
            # Streaming not used here; return full text
            return resp.choices[0].message.content

        # Ollama path with one retry on HPC failover
        try:
            return self._ollama_chat_once(messages, model, temperature, num_ctx, max_tokens)
        except Exception:
            if self.mode == "hpc" and self._cluster:
                # Invalidate and retry once
                try:
                    self._cluster["invalidate"]()
                    return self._ollama_chat_once(messages, model, temperature, num_ctx, max_tokens)
                except Exception as e:
                    raise e
            raise

    def embed(self, text: str, model: Optional[str] = None) -> Optional[List[float]]:
        model = model or self.default_embed_model
        try:
            if self.provider == "openai":
                e = self._client.embeddings.create(model=model, input=text)
                return e.data[0].embedding
            if self.provider == "groq":
                # Groq does not currently expose embeddings; caller should disable memory.
                return None
            # Ollama
            if self.mode == "hpc" and self._cluster:
                client, _ = self._cluster["get"](self.default_model or "llama3.1", force=False)
                res = client.embeddings(model=model, input=text)
                return res["embedding"]
            if self._ollama_client:
                res = self._ollama_client.embeddings(model=model, input=text)
                return res["embedding"]
            res = self._ollama.embeddings(model=model, input=text)
            return res["embedding"]
        except Exception:
            if self.mode == "hpc" and self._cluster:
                try:
                    self._cluster["invalidate"]()
                    client, _ = self._cluster["get"](self.default_model or "llama3.1", force=True)
                    res = client.embeddings(model=model, input=text)
                    return res["embedding"]
                except Exception:
                    return None
            return None
