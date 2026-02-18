import os
import time
import hashlib
import math
from collections import OrderedDict
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import anthropic

app = FastAPI(title="AI Cache API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Config ────────────────────────────────────────────────────────────────────
MAX_CACHE_SIZE = 200
TTL_SECONDS = 24 * 60 * 60       # 24 hours
SEMANTIC_THRESHOLD = 0.82
AVG_TOKENS = 3000
COST_PER_1M = 1.0
BASELINE_DAILY_COST = 8.75

# ── Anthropic client ──────────────────────────────────────────────────────────
client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))

# ── Helpers ───────────────────────────────────────────────────────────────────
def normalize(text: str) -> str:
    import re
    return re.sub(r"[^\w\s]", "", text.lower().strip()).replace(r"\s+", " ")

def md5_key(text: str) -> str:
    return hashlib.md5(normalize(text).encode()).hexdigest()[:8]

def tokenize(text: str) -> list[str]:
    import re
    return re.sub(r"[^\w\s]", "", text.lower()).split()

def cosine_similarity(a: str, b: str) -> float:
    toks_a, toks_b = tokenize(a), tokenize(b)
    vocab = set(toks_a) | set(toks_b)
    if not vocab:
        return 0.0
    vec_a = {w: toks_a.count(w) for w in vocab}
    vec_b = {w: toks_b.count(w) for w in vocab}
    dot = sum(vec_a[w] * vec_b[w] for w in vocab)
    mag_a = math.sqrt(sum(v ** 2 for v in vec_a.values()))
    mag_b = math.sqrt(sum(v ** 2 for v in vec_b.values()))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)

# ── LRU Cache ─────────────────────────────────────────────────────────────────
class LRUCache:
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.store: OrderedDict[str, dict] = OrderedDict()

    def _evict(self):
        while len(self.store) >= self.max_size:
            self.store.popitem(last=False)  # remove oldest

    def set(self, key: str, value: str, query_text: str):
        self._evict()
        self.store[key] = {
            "value": value,
            "query_text": query_text,
            "timestamp": time.time(),
            "hits": 0,
        }
        self.store.move_to_end(key)

    def get(self, key: str) -> Optional[dict]:
        entry = self.store.get(key)
        if not entry:
            return None
        if time.time() - entry["timestamp"] > TTL_SECONDS:
            del self.store[key]
            return None
        entry["hits"] += 1
        self.store.move_to_end(key)
        return entry

    def semantic_search(self, query: str) -> Optional[dict]:
        best, best_sim = None, 0.0
        for key, entry in self.store.items():
            if time.time() - entry["timestamp"] > TTL_SECONDS:
                continue
            sim = cosine_similarity(query, entry["query_text"])
            if sim > best_sim and sim >= SEMANTIC_THRESHOLD:
                best_sim = sim
                best = {"key": key, "entry": entry, "similarity": round(sim, 4)}
        return best

    def size(self) -> int:
        return len(self.store)

    def all_entries(self) -> list[dict]:
        return [
            {"key": k, **v}
            for k, v in reversed(list(self.store.items()))
        ]

cache = LRUCache(MAX_CACHE_SIZE)

# ── Stats ─────────────────────────────────────────────────────────────────────
stats = {
    "total_requests": 0,
    "exact_hits": 0,
    "semantic_hits": 0,
    "misses": 0,
    "tokens_saved": 0,
    "latencies": [],
}

# ── Seed cache ────────────────────────────────────────────────────────────────
SEEDS = [
    ("summarize the quarterly financial report",
     "The quarterly financial report shows revenue growth of 12% YoY, EBITDA margins improved to 23%, and strong cash flow of $4.2M. Key highlights include expansion into two new markets and a 15% reduction in operational costs."),
    ("what are the key points of the privacy policy",
     "The privacy policy covers: data collection practices, how data is shared with third parties, user rights to access/delete data, cookie usage, and contact information for privacy concerns. Data is retained for 2 years after account closure."),
    ("summarize the employee handbook",
     "The employee handbook outlines: company mission and values, work hours and remote work policy (3 days in-office), benefits (health, dental, 401k with 4% match), PTO policy (15 days + 10 holidays), code of conduct, and performance review process."),
]
for q, a in SEEDS:
    cache.set(md5_key(q), a, q)

# ── Models ────────────────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    query: str
    application: str = "document summarizer"

# ── Routes ───────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "ok", "service": "AI Cache API", "version": "1.0.0"}

@app.post("/")
async def query_endpoint(body: QueryRequest):
    start = time.time()
    stats["total_requests"] += 1

    key = md5_key(body.query)
    cache_type = None
    similarity = None

    # 1. Exact match
    entry = cache.get(key)
    if entry:
        answer = entry["value"]
        cached = True
        cache_type = "exact"
        stats["exact_hits"] += 1
        stats["tokens_saved"] += AVG_TOKENS
    else:
        # 2. Semantic match
        sem = cache.semantic_search(body.query)
        if sem:
            answer = sem["entry"]["value"]
            cached = True
            cache_type = "semantic"
            similarity = sem["similarity"]
            stats["semantic_hits"] += 1
            stats["tokens_saved"] += AVG_TOKENS
            cache.set(key, answer, body.query)  # promote under new key
        else:
            # 3. Cache miss → LLM
            stats["misses"] += 1
            try:
                response = client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=1000,
                    system="You are a document summarizer assistant. Provide concise, clear summaries. Keep responses under 150 words.",
                    messages=[{"role": "user", "content": body.query}],
                )
                answer = response.content[0].text
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"LLM error: {str(e)}")
            cache.set(key, answer, body.query)
            cached = False
            cache_type = "miss"

    latency = round((time.time() - start) * 1000)  # ms
    stats["latencies"].append(latency)

    result = {
        "answer": answer,
        "cached": cached,
        "cacheType": cache_type,
        "latency": latency,
        "cacheKey": key,
    }
    if similarity:
        result["similarity"] = similarity
    return result


@app.get("/analytics")
def analytics():
    total = stats["total_requests"]
    hits = stats["exact_hits"] + stats["semantic_hits"]
    hit_rate = round(hits / total, 4) if total > 0 else 0.0
    cost_savings = round((stats["tokens_saved"] * COST_PER_1M) / 1_000_000, 4)
    avg_latency = (
        round(sum(stats["latencies"]) / len(stats["latencies"]))
        if stats["latencies"] else 0
    )

    return {
        "hitRate": hit_rate,
        "totalRequests": total,
        "exactHits": stats["exact_hits"],
        "semanticHits": stats["semantic_hits"],
        "cacheMisses": stats["misses"],
        "cacheSize": cache.size(),
        "costSavings": cost_savings,
        "savingsPercent": round(hit_rate * 100),
        "avgLatencyMs": avg_latency,
        "baselineDailyCost": BASELINE_DAILY_COST,
        "tokensSaved": stats["tokens_saved"],
        "strategies": [
            "exact match (MD5 hash)",
            "semantic similarity (TF-IDF cosine ≥ 0.82)",
            "LRU eviction (max 200 entries)",
            "TTL expiration (24 hours)",
        ],
    }


@app.get("/cache")
def cache_state():
    return {
        "size": cache.size(),
        "maxSize": MAX_CACHE_SIZE,
        "ttlSeconds": TTL_SECONDS,
        "entries": cache.all_entries(),
    }


@app.delete("/cache")
def clear_cache():
    cache.store.clear()
    return {"cleared": True, "size": 0}
