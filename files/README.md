# AI Cache API — Render Deployment

## Files
- `main.py` — FastAPI app with exact + semantic caching, LRU eviction, TTL
- `requirements.txt` — Python dependencies
- `render.yaml` — Render auto-deploy config

## Deploy to Render (5 steps)

1. **Push to GitHub**
   ```bash
   git init
   git add .
   git commit -m "initial"
   git remote add origin https://github.com/YOUR_USERNAME/ai-cache-api.git
   git push -u origin main
   ```

2. **Create Web Service on Render**
   - Go to https://render.com → New → Web Service
   - Connect your GitHub repo
   - Render auto-detects `render.yaml` — click **Apply**

3. **Add Environment Variable**
   - In Render dashboard → Environment → Add:
     - Key: `ANTHROPIC_API_KEY`
     - Value: your Anthropic API key (from https://console.anthropic.com)

4. **Deploy** — Render builds and starts automatically (~2 min)

5. **Your URLs**
   ```
   POST  https://ai-cache-api.onrender.com/          ← submit this
   GET   https://ai-cache-api.onrender.com/analytics
   GET   https://ai-cache-api.onrender.com/cache
   ```

## API Usage

### POST /
```json
{
  "query": "summarize the quarterly financial report",
  "application": "document summarizer"
}
```
Response:
```json
{
  "answer": "...",
  "cached": true,
  "cacheType": "exact",
  "latency": 12,
  "cacheKey": "a3f9c2b1"
}
```

### GET /analytics
Returns hit rate, cost savings, cache size, and active strategies.

## Caching Strategies Implemented
| Strategy | Description |
|---|---|
| Exact match | MD5 hash of normalized query |
| Semantic match | TF-IDF cosine similarity ≥ 0.82 |
| LRU eviction | Max 200 entries, oldest removed first |
| TTL expiration | Entries expire after 24 hours |
