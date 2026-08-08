# ──────────────────────────────────────────────────────────────────────────────
# Improved Hybrid Memory
# ──────────────────────────────────────────────────────────────────────────────
from dataclasses import dataclass, asdict
import tempfile
import math
import numpy as np
import os
import time
import json
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import uuid
from llm_provider import LLMProvider

try:
    import tiktoken
except ImportError:
    tiktoken = None


@dataclass
class Episode:
    id: str
    iteration: int
    policy_name: str
    policy_desc: str
    avg_cache_hit_rate: float
    avg_ipc: float
    per_workload: Dict[str, Dict[str, float]]  
    compile_ok: bool
    lesson: str  
    ts: float
    embedding: Optional[List[float]] = None
    lineage_id: Optional[str] = None
    code_path: Optional[str] = None
    code_hash: Optional[str] = None

class HybridMemory:
    """
    - Compresses large turns on add (never store full code).
    - Keeps episodic results with embeddings for retrieval.
    - Maintains a compact long-term summary (principles/don’ts/knobs).
    - Token-budget-aware message builder with retrieval.
    """
    def __init__(
            self,
            llm: LLMProvider,
            model: str,
            memory_path: Path,
            max_tokens: int = 7000,
            ephemeral_max_tokens: int = 1600,
            summary_max_tokens: int = 1200,
            compress_threshold_tokens: int = 900,
            retrieve_k: int = 5,
            use_embeddings: bool = True,
            memory_model: str | None = None,
            ):
        
        self.llm = llm
        self.model = model
        self.memory_path = memory_path
        self.max_tokens = max_tokens
        self.ephemeral_max_tokens = ephemeral_max_tokens
        self.summary_max_tokens = summary_max_tokens
        self.compress_threshold_tokens = compress_threshold_tokens
        self.retrieve_k = retrieve_k
        self.use_embeddings = use_embeddings
        self.memory_model = memory_model

        self.chat: List[Dict[str, str]] = [] 
        self.summary: Dict[str, List[str]] = {  
            "principles": [],
            "donts": [],
            "knobs": [],
            "open_questions": []
        }
        self.episodes: List[Episode] = []

        self._encoding = None
        self._load()
        self._init_encoding()

    # ── Token helpers ─────────────────────────────────────────────────────────
    def _init_encoding(self):
        if tiktoken:
            try:
                self._encoding = tiktoken.encoding_for_model(self.model)
            except Exception:
                self._encoding = tiktoken.get_encoding("cl100k_base")

    def _count_tokens_text(self, text: str) -> int:
        if not text:
            return 0
        if self._encoding:
            return len(self._encoding.encode(text))
        # fallback heuristic
        return max(1, int(len(text) / 4))

    def _count_tokens_messages(self, msgs: List[Dict[str, str]]) -> int:
        # rough but consistent measurement
        return sum(self._count_tokens_text(m.get("content", "")) + 6 for m in msgs)

    # ── Persistence ───────────────────────────────────────────────────────────
    def _load(self):
        if self.memory_path.exists():
            try:
                data = json.loads(self.memory_path.read_text(encoding="utf-8"))
                self.chat = data.get("chat", [])
                self.summary = data.get("summary", self.summary)
                self.episodes = [Episode(**e) for e in data.get("episodes", [])]
            except Exception:
                self.chat, self.episodes = [], []

    def _save(self):
        payload = {
            "chat": self.chat,
            "summary": self.summary,
            "episodes": [asdict(e) for e in self.episodes],
        }
        tmp = self.memory_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.memory_path)

    # ── LLM helpers ───────────────────────────────────────────────────────────
    def _compress_content(self, label: str, content: str) -> str:
        try:
            prompt = (
                "Summarize the following cache policy design turn into ≤12 bullets, "
                "capturing only: core ideas, heuristics, tunable parameters, and decisions. "
                "Omit code and boilerplate. Keep it ≤180 tokens.\n\n"
                f"Label: {label}\n---\n{content}\n---"
            )
            resp_text = self.llm.chat(
                messages=[{"role": "user", "content": prompt}],
                model=(self.memory_model or self.model),
                temperature=0.1,
                num_ctx=4096
            )
            return resp_text.strip()
        except Exception:
            return content[:1500]

    def _summarize_memory_with_new_fragments(self, fragments: List[str]):
        """
        Update structured long-term memory from fragments + current summary.
        """
        try:
            existing = json.dumps(self.summary, ensure_ascii=False)
            joined = "\n\n".join(fragments)
            sys = (
                "You maintain a compact, structured long-term memory for cache policy discovery.\n"
                "Update the JSON fields (principles, donts, knobs, open_questions) using new fragments.\n"
                "Be terse, non-redundant. Keep each list to ≤8 items. Output ONLY JSON."
            )
            user = f"Existing JSON:\n{existing}\n\nNew fragments:\n{joined}"
            resp_text = self.llm.chat(
                messages=[{"role":"system","content": sys}, {"role":"user","content": user}],
                model=(self.memory_model or self.model),
                temperature=0.1,
                num_ctx=4096
            )
            data = json.loads(resp_text.strip())
            for k in ["principles", "donts", "knobs", "open_questions"]:
                if k in data and isinstance(data[k], list):
                    data[k] = data[k][:8]
            self.summary = data
        except Exception:
            # fallback: append as plain bullets into principles
            for frag in fragments:
                self.summary["principles"].append(f"- {frag[:180]}")
            self.summary["principles"] = self.summary["principles"][-8:]

        # Enforce token limit on textual rendering of summary
        text_render = self._render_summary_text()
        if self._count_tokens_text(text_render) > self.summary_max_tokens:
            # compress the whole summary textual rendering
            compressed = self._compress_content("summary", text_render)
            # Try to split compressed into bullets again
            bullets = [b.strip("- ") for b in compressed.split("\n") if b.strip()]
            self.summary = {
                "principles": bullets[:6],
                "donts": bullets[6:10],
                "knobs": bullets[10:14],
                "open_questions": bullets[14:18],
            }

    def _embed(self, text: str) -> Optional[List[float]]:
        return self.llm.embed(text, model=None)  # provider default embedding model

    # ── Public API ────────────────────────────────────────────────────────────
    def lineage_history(self, lineage_id: str, k: int = 5) -> List[Episode]:
        eps = [e for e in self.episodes if e.lineage_id == lineage_id]
        eps.sort(key=lambda e: e.iteration, reverse=True)
        return eps[:k]

    def render_lineage_block(
        self,
        lineage_id: str,
        k: int = 5,
        include_desc: bool = True,
        desc_max_chars: int = 180
    ) -> str:
        """
        Render a compact summary of the last k variants in this lineage, optionally
        including a truncated one-line description for each.

        Example:
        Variants tested so far for this policy lineage (most recent first):
        - iter=12 name=pc_rrip_v3 avg_ipc=1.821500 avg_hr=0.452100 compile_ok=True
        desc: Prioritize demand hits; reduce prefetch pollution via higher RRPV insertions.
        - iter=11 name=pc_rrip_v2 avg_ipc=1.799200 avg_hr=0.447800 compile_ok=True
        desc: Tweak insertion priority for PC groups; mild bypass for cold PCs.
        Best so far: pc_rrip_v3 at iter 12 (avg_ipc=1.821500)
        """
        def _oneline(s: str) -> str:
            return " ".join((s or "").split())

        def _truncate(s: str, n: int) -> str:
            if len(s) <= n:
                return s
            # try to cut at a space before the limit
            cut = s.rfind(" ", 0, max(0, n - 3))
            if cut == -1:
                cut = n - 3
            return s[:cut].rstrip() + "..."

        eps = self.lineage_history(lineage_id, k)
        if not eps:
            return ""

        lines = ["Variants tested so far for this policy lineage (most recent first):"]
        for e in eps:
            lines.append(
                f"- iter={e.iteration} name={e.policy_name} avg_ipc={e.avg_ipc:.6f} "
                f"avg_hr={e.avg_cache_hit_rate:.6f} compile_ok={e.compile_ok}"
            )
            if include_desc and getattr(e, "policy_desc", None):
                desc = _truncate(_oneline(e.policy_desc), desc_max_chars)
                lines.append(f"  desc: {desc}")

        best = max(eps, key=lambda x: x.avg_ipc, default=None)
        if best:
            lines.append(f"Best so far: {best.policy_name} at iter {best.iteration} (avg_ipc={best.avg_ipc:.6f})")

        return "\n".join(lines)

    def add_chat(self, role: str, content: str, label: Optional[str] = None):
        """
        Add a chat turn; compress if large. Never store raw code here.
        """
        label = label or role
        if self._count_tokens_text(content) > self.compress_threshold_tokens:
            content = self._compress_content(label, content)
        self.chat.append({"role": role, "content": content, "label": label, "ts": time.time()})
        self._trim_ephemeral()
        self._save()

    def record_experiment(
        self,
        iteration: int,
        policy_name: str,
        policy_desc: str,
        results: List[Tuple[str, float, float]],  # (workload, hit_rate, ipc)
        compile_ok: bool,
        lineage_id: Optional[str] = None,
        code_path: Optional[str] = None,
        code_hash: Optional[str] = None,
    ):
        per_wl = {w: {"hit_rate": hr, "ipc": ipc} for (w, hr, ipc) in results if hr is not None}
        avg_hit = sum(v["hit_rate"] for v in per_wl.values()) / max(1, len(per_wl))
        avg_ipc = sum(v["ipc"] for v in per_wl.values()) / max(1, len(per_wl))
        lesson = f"{policy_name}: {policy_desc[:220]} | avg_hit={avg_hit:.6f}, avg_ipc={avg_ipc:.6f}; " \
                f"best={max(per_wl, key=lambda k: per_wl[k]['hit_rate']) if per_wl else 'n/a'}"
        emb = self._embed(f"{policy_name}. {policy_desc}. {lesson}")
        ep = Episode(
            id=str(uuid.uuid4()),
            iteration=iteration,
            policy_name=policy_name,
            policy_desc=policy_desc,
            avg_cache_hit_rate=avg_hit,
            avg_ipc=avg_ipc,
            per_workload=per_wl,
            compile_ok=compile_ok,
            lesson=lesson,
            ts=time.time(),
            embedding=emb,
            lineage_id=lineage_id,
            code_path=code_path,
            code_hash=code_hash
        )
        self.episodes.append(ep)
        self._summarize_memory_with_new_fragments([policy_desc, lesson])
        self._save()

    def build_messages(self, system_prompt: str, user_prompt: str, retrieve_k: Optional[int] = None) -> List[Dict[str, str]]:
        retrieve_k = retrieve_k or self.retrieve_k
        msgs: List[Dict[str, str]] = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})

        # Long-term memory (structured) as short text
        ltm = self._render_summary_text()
        if ltm:
            msgs.append({"role": "system", "content": "Long-term memory:\n" + ltm})

        # Retrieval: pick top-k episodes by cosine similarity
        retrieved = self._retrieve_episodes(user_prompt, top_k=retrieve_k)
        if retrieved:
            block = "Relevant prior lessons:\n" + "\n".join(
                f"- [{e.iteration}] {e.policy_name}: {e.lesson}" for e in retrieved
            )
            msgs.append({"role": "system", "content": block})

        # Recent chat context (already trimmed to budget)
        msgs.extend({"role": m["role"], "content": m["content"]} for m in self.chat[-8:])

        # Current instruction
        msgs.append({"role": "user", "content": user_prompt})

        # If we still exceed the total budget, trim recent chat first
        self._enforce_total_budget(msgs)
        return msgs

    # ── Internals ─────────────────────────────────────────────────────────────
    def _render_summary_text(self) -> str:
        def section(title, items):
            if not items:
                return ""
            return title + "\n" + "\n".join(f"- {x}" for x in items) + "\n"
        return (
            section("Principles", self.summary.get("principles", [])) +
            section("Don'ts", self.summary.get("donts", [])) +
            section("Knobs", self.summary.get("knobs", [])) +
            section("Open Questions", self.summary.get("open_questions", []))
        ).strip()

    def _retrieve_episodes(self, query: str, top_k: int) -> List[Episode]:
        if not self.use_embeddings:
            # simple fallback: last K
            return list(sorted(self.episodes, key=lambda e: e.ts, reverse=True))[:top_k]
        q_emb = self._embed(query)
        if not q_emb:
            return list(sorted(self.episodes, key=lambda e: e.ts, reverse=True))[:top_k]
        q = np.array(q_emb, dtype=np.float32)
        scored = []
        for e in self.episodes:
            if e.embedding:
                v = np.array(e.embedding, dtype=np.float32)
                denom = (np.linalg.norm(q) * np.linalg.norm(v)) or 1.0
                sim = float(np.dot(q, v) / denom)
                scored.append((sim, e))
        scored.sort(key=lambda t: t[0], reverse=True)
        return [e for _, e in scored[:top_k]]

    def _trim_ephemeral(self):
        # Keep recent chat under the ephemeral budget
        while self._count_tokens_messages([{"role": m["role"], "content": m["content"]} for m in self.chat]) > self.ephemeral_max_tokens:
            if self.chat:
                self.chat.pop(0)
            else:
                break

    def _enforce_total_budget(self, msgs: List[Dict[str, str]]):
        # If total exceeds, first drop oldest chat (they appear before user prompt)
        while self._count_tokens_messages(msgs) > self.max_tokens:
            # Remove the first non-system message except final user message
            idx = next((i for i, m in enumerate(msgs) if m["role"] != "system" and i < len(msgs) - 1), None)
            if idx is None:
                break
            msgs.pop(idx)

    def add_event(self, kind: str, details: str, severity: str = "info"):
        """
        Record a concise, labeled system event that the next iteration should see.
        - kind: 'parse_error' | 'format_violation' | 'compile_error' | 'runtime_error' | 'regression' | 'improvement'
        - severity: 'info' | 'warn' | 'error'
        """
        # Keep it short and on one line if possible
        msg = f"[{kind.upper()}][{severity}] " + details.strip().replace("\n", " ")
        msg = msg[:500]  # hard cap
        self.chat.append({"role": "system", "content": msg, "label": kind, "ts": time.time()})

        # Nudge long-term memory with useful “don’ts” for hard failures
        if kind in {"format_violation", "compile_error"}:
            first = details.strip().splitlines()[0] if details else kind
            dont = f"Avoid {first[:110]}"
            donts = self.summary.get("donts", [])
            donts.append(dont)
            self.summary["donts"] = donts[-8:]  # cap length

        self._trim_ephemeral()
        self._save()