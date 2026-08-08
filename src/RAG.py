import sqlite3
from typing import List, Dict, Tuple, Optional
import common
import re

metric_aliases = {
    # IPC / score
    "ipc": "IPC",
    "avg ipc": "IPC",
    "average ipc": "IPC",
    "score": "score",
    "efficiency": "score",
    "ipc per kb": "score",
    "ipc per kib": "score",
    "ipc_per_kb": "score",
    # Total (cache) hit rate
    "total": "cache_hit_rate",
    "hr": "cache_hit_rate",
    "hit rate": "cache_hit_rate",
    "hitrate": "cache_hit_rate",
    "total hit rate": "cache_hit_rate",
    "total hr": "cache_hit_rate",
    "cache hit rate": "cache_hit_rate",
    # Per-category hit rates
    "load": "load_hit_rate",
    "load hr": "load_hit_rate",
    "load hit rate": "load_hit_rate",
    "rfo": "rfo_hit_rate",
    "rfo hr": "rfo_hit_rate",
    "rfo hit rate": "rfo_hit_rate",
    "prefetch": "prefetch_hit_rate",
    "pref": "prefetch_hit_rate",
    "prefetch hr": "prefetch_hit_rate",
    "prefetch hit rate": "prefetch_hit_rate",
    "writeback": "writeback_hit_rate",
    "write back": "writeback_hit_rate",
    "write-back": "writeback_hit_rate",
    "wb": "writeback_hit_rate",
    "writeback hr": "writeback_hit_rate",
    "writeback hit rate": "writeback_hit_rate",
    # Storage
    "storage": "storage_kb",
    "storage kb": "storage_kb",
    "storage_kb": "storage_kb",
    "area": "storage_kb",
    # Optional derived metric (handle specially)
    "miss rate": "__MISS_RATE__",
    "missrate": "__MISS_RATE__",
}

def normalize_metric_key(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[\s\-_]+", " ", s)  # collapse spaces/hyphens/underscores
    s = s.replace("(", "").replace(")", "")  # remove parentheses
    return s

def resolve_metric(metric: str) -> tuple[str, bool, str]:
    """
    Returns (column_name, descending, display_label).
    For derived metrics, maps to an actual column and adjusts sort direction.
    """
    key = normalize_metric_key(metric)
    col = metric_aliases.get(key)
    if not col:
        raise ValueError(f"Unknown metric '{metric}'. Allowed keys include: {sorted(metric_aliases.keys())}")

    # Default: bigger is better
    descending = True
    # Pretty label for printing
    display_label = key.upper()

    if col == "storage_kb":
        descending = False  # smaller is better
        display_label = "STORAGE_KB"

    if col == "__MISS_RATE__":
        # Miss rate = 1 - hit rate => sort by hit rate ascending
        col = "cache_hit_rate"
        descending = False
        display_label = "MISS RATE"

    if col == "IPC":
        display_label = "IPC"
    elif col == "cache_hit_rate":
        display_label = "TOTAL HIT RATE"
    elif col == "load_hit_rate":
        display_label = "LOAD HIT RATE"
    elif col == "rfo_hit_rate":
        display_label = "RFO HIT RATE"
    elif col == "prefetch_hit_rate":
        display_label = "PREFETCH HIT RATE"
    elif col == "writeback_hit_rate":
        display_label = "WRITEBACK HIT RATE"
    elif col == "score":
        display_label = "SCORE"

    return col, descending, display_label

class ExperimentRAG:
    
    def __init__(self, db_path: str = 'funsearch.db'):
        """Initialize the RAG system with database connection"""
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()

    def close(self):
        """Close database connection"""
        self.conn.close()

    def get_all_workloads_with_description(self) -> str:
        """
        Retrieve all workloads and their descriptions from workloads table.

        Returns string:
          "workload1: description1\nworkload2: description2\n..."
        """
        query = '''
        SELECT name, description
        FROM workloads
        ORDER BY name
        '''
        self.cursor.execute(query)
        rows = self.cursor.fetchall()
        return '\n'.join(f"{name}: {desc}" for name, desc in rows)

    def print_top_policies_by_metric(
        self,
        workload: str,
        metric: str = "ipc",
        top_n: int = 5,
        descending: Optional[bool] = None
    ) -> List[Dict]:
        # Resolve metric and default sort direction
        col, default_desc, label = resolve_metric(metric)
        order_desc = default_desc if descending is None else bool(descending)
        order_dir = "DESC" if order_desc else "ASC"

        # Optional: push NULL storage to the end when sorting by storage
        null_clause = ""
        if col == "storage_kb":
            null_clause = "CASE WHEN e.storage_kb IS NULL THEN 1 ELSE 0 END, "

        query = f"""
            SELECT
                e.policy,
                e.policy_description,
                COALESCE(w.description, '') AS workload_description,
                e.cpp_file_path,
                e.IPC,
                e.cache_hit_rate,
                e.load_hit_rate,
                e.rfo_hit_rate,
                e.prefetch_hit_rate,
                e.writeback_hit_rate,
                e.score,
                e.storage_kb,
                e.storage_note
            FROM experiments AS e
            LEFT JOIN workloads AS w ON w.name = e.workload
            WHERE e.workload = ?
            ORDER BY {null_clause}{col} {order_dir}
            LIMIT ?
        """
        self.cursor.execute(query, (workload, top_n))
        rows = self.cursor.fetchall()

        results: List[Dict] = []
        for row in rows:
            results.append({
                "policy": row[0],
                "policy_description": row[1],
                "workload_description": row[2],
                "cpp_file_path": row[3],
                "ipc": float(row[4]),
                "total_hit_rate": float(row[5]),
                "load_hit_rate": float(row[6]),
                "rfo_hit_rate": float(row[7]),
                "prefetch_hit_rate": float(row[8]),
                "writeback_hit_rate": float(row[9]),
                "score": float(row[10]),
                "storage_kb": (None if row[11] is None else float(row[11])),
                "storage_note": row[12] or "",
            })

        print(f"\nTop {len(results)} policies by {label} for workload: {workload}")
        for i, r in enumerate(results, 1):
            metric_value = (
                r["ipc"] if col == "IPC" else
                r["score"] if col == "score" else
                r["total_hit_rate"] if col == "cache_hit_rate" else
                r["load_hit_rate"] if col == "load_hit_rate" else
                r["rfo_hit_rate"] if col == "rfo_hit_rate" else
                r["prefetch_hit_rate"] if col == "prefetch_hit_rate" else
                r["writeback_hit_rate"] if col == "writeback_hit_rate" else
                r["storage_kb"]
            )
            if col in {"IPC", "score", "storage_kb"}:
                metric_str = "n/a" if metric_value is None else f"{metric_value:.6f}"
            else:
                metric_str = f"{metric_value:.6f}"

            storage_str = "n/a" if r["storage_kb"] is None else f"{r['storage_kb']:.2f} KB"
            storage_note = f" ({r['storage_note']})" if r.get("storage_note") else ""

            print(f"{i}. {r['policy']}  {label}={metric_str}")
            print(f"   IPC={r['ipc']:.6f}  TOTAL={r['total_hit_rate']:.6f}  "
                  f"LOAD={r['load_hit_rate']:.6f}  RFO={r['rfo_hit_rate']:.6f}  "
                  f"PREF={r['prefetch_hit_rate']:.6f}  WB={r['writeback_hit_rate']:.6f}")
            print(f"   Score={r['score']:.6f} ({common.SCORE})")
            print(f"   Storage={storage_str}{storage_note}")
            print(f"   {r['policy_description']}")
            print(f"   {r['cpp_file_path']}\n")

        return results

    def get_top_by_metric(
        self,
        workload: str,
        metric: str = "ipc",
        top_n: int = 2,
        descending: Optional[bool] = None
    ) -> List[Dict]:
        col, default_desc, _label = resolve_metric(metric)
        order_desc = default_desc if descending is None else bool(descending)
        order_dir = "DESC" if order_desc else "ASC"

        null_clause = ""
        if col == "storage_kb":
            null_clause = "CASE WHEN e.storage_kb IS NULL THEN 1 ELSE 0 END, "

        query = f"""
            SELECT e.policy, e.policy_description, COALESCE(w.description, '') AS workload_description,
                   e.cpp_file_path, e.IPC, e.cache_hit_rate, e.load_hit_rate, e.rfo_hit_rate,
                   e.prefetch_hit_rate, e.writeback_hit_rate, e.score, e.storage_kb, e.storage_note
            FROM experiments AS e
            LEFT JOIN workloads AS w ON w.name = e.workload
            WHERE e.workload = ?
            ORDER BY {null_clause}{col} {order_dir}
            LIMIT ?
        """
        self.cursor.execute(query, (workload, top_n))
        rows = self.cursor.fetchall()
        results: List[Dict] = []
        for row in rows:
            results.append({
                "policy": row[0],
                "policy_description": row[1],
                "workload_description": row[2],
                "cpp_file_path": row[3],
                "ipc": float(row[4]),
                "total_hit_rate": float(row[5]),
                "load_hit_rate": float(row[6]),
                "rfo_hit_rate": float(row[7]),
                "prefetch_hit_rate": float(row[8]),
                "writeback_hit_rate": float(row[9]),
                "score": float(row[10]),
                "storage_kb": (None if row[11] is None else float(row[11])),
                "storage_note": row[12] or "",
            })
        return results
