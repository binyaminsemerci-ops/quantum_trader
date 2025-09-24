"""Minimal Twitter/X client wrapper.

Environment variables supported:
- X_BEARER_TOKEN: preferred (App-only bearer token for v2 endpoints)
- X_API_KEY, X_API_SECRET, X_ACCESS_TOKEN, X_ACCESS_SECRET: optional OAuth1

This client is intentionally small and resilient: when no credentials are
configured it returns a neutral sentiment. It provides a simple `sentiment_for_symbol`
method that queries recent public tweets mentioning the symbol and computes a
very small heuristic sentiment score.
"""
from typing import Optional, Dict, Any, List
import os
import time
import requests
import math
import warnings

# Small in-process cache to avoid repeated API calls during short tests
_CACHE: Dict[str, Any] = {}


class TwitterClient:
    """Robust, minimal Twitter/X client for sentiment heuristics.

    Behavior and features:
    - Uses X_BEARER_TOKEN when available. Falls back to mock if not configured.
    - Retries transient errors (5xx, 429) with exponential backoff.
    - Simple in-memory TTL cache to limit repeated calls.
    - Small, explainable sentiment heuristic (word counting) to avoid heavy NLP deps.
    """

    def __init__(self):
        self.bearer = os.getenv('X_BEARER_TOKEN')
        # Optionally support OAuth1 env vars in the future
        self.mock = not bool(self.bearer)
        self.base_v2 = 'https://api.twitter.com/2'

    def _cached(self, key: str, ttl: int = 30):
        e = _CACHE.get(key)
        if not e:
            return None
        ts, val = e
        if time.time() - ts > ttl:
            del _CACHE[key]
            return None
        return val

    def _set_cache(self, key: str, val: Any):
        _CACHE[key] = (time.time(), val)

    def _request_with_retries(self, url: str, params: dict, headers: dict, max_attempts: int = 3, timeout: int = 6):
        attempt = 0
        while attempt < max_attempts:
            try:
                r = requests.get(url, params=params, headers=headers, timeout=timeout)
                if r.status_code == 200:
                    return r
                # If rate limited or server error, back off and retry
                if r.status_code in (429, 500, 502, 503, 504):
                    backoff = (2 ** attempt) + (0.1 * attempt)
                    time.sleep(backoff)
                    attempt += 1
                    continue
                # other client errors: return response for caller to inspect
                return r
            except requests.RequestException:
                # network blip: backoff and retry
                backoff = (2 ** attempt) + 0.1
                time.sleep(backoff)
                attempt += 1
                continue
        return None

    def sentiment_for_symbol(self, symbol: Optional[str] = None, max_results: int = 20) -> Dict[str, Any]:
        """Return a lightweight sentiment summary for a symbol.

        Args:
            symbol: e.g. 'BTC' or 'ETH'. If None, a general crypto query is used.
            max_results: how many recent tweets to fetch (bounded by API limits).

        Returns:
            dict with keys: score (float), label (str), source (twitter|mock|error), and optional code.
        """
        if self.mock:
            return {'score': 0.0, 'label': 'neutral', 'source': 'mock'}

        q = (symbol or 'crypto') + ' -is:retweet lang:en'
        cache_key = f'tw:{q}:{max_results}'
        cached = self._cached(cache_key, ttl=30)
        if cached:
            return cached

        headers = {'Authorization': f'Bearer {self.bearer}'}
        # Twitter v2 recent search allows max_results up to 100 depending on access tier
        params = {'query': q, 'max_results': min(100, max_results), 'tweet.fields': 'text,created_at'}
        url = f'{self.base_v2}/tweets/search/recent'

        r = self._request_with_retries(url, params=params, headers=headers, max_attempts=4, timeout=8)
        if r is None:
            warnings.warn('Twitter API request failed after retries')
            return {'score': 0.0, 'label': 'neutral', 'source': 'error'}

        if r.status_code != 200:
            return {'score': 0.0, 'label': 'neutral', 'source': 'error', 'code': r.status_code}

        data = r.json()
        texts = [t.get('text', '') for t in data.get('data', [])]

        # trivial, deterministic heuristic: count positive/negative tokens
        pos_words = ['good', 'great', 'bull', 'moon', 'buy', 'bullish', 'gain', 'pump']
        neg_words = ['bad', 'sell', 'bear', 'down', 'dump', 'loss', 'crash']
        score = 0
        for txt in texts:
            low = txt.lower()
            for w in pos_words:
                if w in low:
                    score += 1
            for w in neg_words:
                if w in low:
                    score -= 1

        # normalize to range roughly between -1.0 and 1.0
        if texts:
            denom = max(1, len(texts) * 2)
            score = score / denom
            # clamp
            score = max(-1.0, min(1.0, score))
        else:
            score = 0.0

        out = {'score': float(score), 'label': 'positive' if score > 0 else 'negative' if score < 0 else 'neutral', 'source': 'twitter'}
        self._set_cache(cache_key, out)
        return out

    def global_sentiment(self) -> Dict[str, Any]:
        return self.sentiment_for_symbol(None)
