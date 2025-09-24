"""Minimal CryptoPanic client wrapper.

Environment variables:
- CRYPTOPANIC_KEY

This wrapper fetches latest posts (public) filtered by symbol tag when possible.
If no key is configured it returns an empty list / mock items.
"""
from typing import Optional, List, Dict, Any
import os
import requests  # type: ignore[import-untyped]
import time
import warnings

_CACHE: dict[str, Any] = {}


class CryptoPanicClient:
    """Minimal but more robust CryptoPanic client.

    - Uses CRYPTOPANIC_KEY when available. Falls back to mock items otherwise.
    - Adds retries/backoff for transient errors and simple caching.
    """

    def __init__(self):
        self.key = os.getenv('CRYPTOPANIC_KEY')
        self.base = 'https://cryptopanic.com/api/v1'
        self.mock = not bool(self.key)

    def _cached(self, key: str, ttl: int = 60):
        v = _CACHE.get(key)
        if not v:
            return None
        ts, val = v
        if time.time() - ts > ttl:
            del _CACHE[key]
            return None
        return val

    def _set_cache(self, key: str, val: Any):
        _CACHE[key] = (time.time(), val)

    def _request_with_retries(self, url: str, params: dict, max_attempts: int = 3, timeout: int = 6):
        attempt = 0
        while attempt < max_attempts:
            try:
                r = requests.get(url, params=params, timeout=timeout)
                if r.status_code == 200:
                    return r
                if r.status_code in (429, 500, 502, 503, 504):
                    sleep_for = (2 ** attempt) + 0.1
                    time.sleep(sleep_for)
                    attempt += 1
                    continue
                return r
            except requests.RequestException:
                time.sleep(0.2 + attempt)
                attempt += 1
                continue
        return None

    def fetch_latest(self, tag: Optional[str] = None, limit: int = 20) -> List[Dict[str, Any]]:
        if self.mock:
            # return deterministic mock items
            now = int(time.time())
            items = []
            for i in range(limit):
                items.append({'id': f'mock-{i}', 'title': f'Mock news {i} for {tag}', 'published_at': now - i * 60})
            return items

        cache_key = f'cp:{tag}:{limit}'
        cached = self._cached(cache_key)
        if cached:
            return cached

        params = {'auth_token': self.key, 'kind': 'news'}
        if tag:
            # cryptopanic's API supports 'filter' parameter for tags; keep safe
            params['filter'] = tag

        r = self._request_with_retries(f'{self.base}/posts/', params=params, max_attempts=3, timeout=8)
        if r is None:
            warnings.warn('CryptoPanic request failed after retries')
            return []

        if r.status_code != 200:
            return []

        try:
            data = r.json()
            items = data.get('results', [])[:limit]
            self._set_cache(cache_key, items)
            return items
        except Exception:
            return []
