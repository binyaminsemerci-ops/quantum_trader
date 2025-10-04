from typing import List, Dict, Any, Optional, Mapping
import os
import pickle
import logging
import asyncio
import numpy as np


from backend.utils.twitter_client import TwitterClient

logger = logging.getLogger(__name__)


class XGBAgent:
    """Lightweight agent wrapper that loads model artifacts and provides
    synchronous and asynchronous helpers for scoring symbols.

    The implementation avoids hard failures when optional deps (pandas, xgboost)
    are missing and uses small fallbacks so endpoints remain responsive.
    """

    def __init__(
        self, model_path: Optional[str] = None, scaler_path: Optional[str] = None
    ):
        base = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
        self.model_path = model_path or os.path.join(base, "xgb_model.pkl")
        self.scaler_path = scaler_path or os.path.join(base, "scaler.pkl")
        self.model = None
        self.scaler = None
        # helper clients
        self.twitter: Optional[TwitterClient] = None
        try:
            self.twitter = TwitterClient()
        except Exception as e:
            logger.debug("Failed to init Twitter client: %s", e)
            self.twitter = None
        self._load()

    def _load(self) -> None:
        """Load model and scaler from disk if present. Swallow errors and log them."""
        try:
            if os.path.exists(self.model_path):
                with open(self.model_path, "rb") as f:
                    self.model = pickle.load(f)
                    logger.info("Loaded model from %s", self.model_path)
        except Exception as e:
            logger.debug("Failed to load model: %s", e)
            self.model = None

        try:
            if os.path.exists(self.scaler_path):
                with open(self.scaler_path, "rb") as f:
                    self.scaler = pickle.load(f)
                    logger.info("Loaded scaler from %s", self.scaler_path)
        except Exception as e:
            logger.debug("Failed to load scaler: %s", e)
            self.scaler = None

    def _features_from_ohlcv(self, df) -> Any:
        """Turn raw OHLCV (DataFrame or list-of-dicts) into model features (last row).

        Raises if pandas or the feature engineer cannot be imported.
        """
        try:
            import pandas as _pd  # type: ignore[import-untyped]

            # local import for feature engineer; mypy in CI may not resolve ai_engine package here
            from ai_engine.feature_engineer import add_technical_indicators as _add_technical_indicators  # type: ignore[import-not-found, import-untyped]
            from ai_engine.feature_engineer import add_sentiment_features as _add_sentiment_features  # type: ignore[import-not-found, import-untyped]
        except Exception:
            logger.debug("Pandas or feature_engineer not available")
            raise

        # Accept both DataFrame and list-of-dicts
        if isinstance(df, list):
            df = _pd.DataFrame(df)

        # Normalize column casing to predictable names
        df.columns = [str(c).lower() for c in df.columns]

        # ensure we have required columns (open, high, low, close, volume)
        for col in ("open", "high", "low", "close", "volume"):
            if col not in df.columns:
                df[col] = _pd.NA

        # prepare DataFrame expected by feature engineer (capitalized names sometimes expected)
        df_norm = df.rename(
            columns={
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
            }
        )

        feat = _add_technical_indicators(df_norm)

        # mypy/pandas: some CI environments use installed pandas stubs while
        # local static analysis may treat DataFrame-like objects as 'object'.
        # We previously cast to Any to help static checkers; drop the unused
        # local assignment here to satisfy linters. Callers that need a
        # dynamic view can cast later when required.

        # sentiment/news may be provided as columns (sentiment, news_count)
        sentiment_series = None
        news_counts = None
        if "sentiment" in df.columns:
            sentiment_series = df["sentiment"]
        if "news_count" in df.columns:
            news_counts = df["news_count"]

        if sentiment_series is not None or news_counts is not None:
            feat = _add_sentiment_features(
                feat, sentiment_series=sentiment_series, news_counts=news_counts
            )

        if feat.shape[0] == 0:
            raise RuntimeError("No features generated")

        # return last row as DataFrame-like object; callers will handle numeric
        # selection defensively. Use Any to avoid strict pandas typing issues here.
        return feat.iloc[-1:]

    def predict_for_symbol(self, ohlcv) -> Dict[str, Any]:
        """Synchronous predict helper. Returns {'action':..., 'score':...}."""
        try:
            feat = self._features_from_ohlcv(ohlcv)
        except Exception as e:
            logger.debug("Feature extraction failed: %s", e)
            return {"action": "HOLD", "score": 0.0}

        # Cast to Any to satisfy static checkers that may not have pandas stubs
        from typing import Any as _Any, cast as _cast

        feat_any = _cast(_Any, feat)

        # select numeric features
        try:
            X = feat_any.select_dtypes(include=[np.number]).to_numpy().astype(float)
        except Exception:
            logger.debug("Failed to select numeric features")
            return {"action": "HOLD", "score": 0.0}

        if X.size == 0:
            return {"action": "HOLD", "score": 0.0}

        # Ensure 2D
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # apply scaler when available
        if self.scaler is not None:
            try:
                Xs = self.scaler.transform(X)
            except Exception as e:
                logger.debug("Scaler.transform failed: %s", e)
                Xs = X
        else:
            Xs = X

        # If no trained model, use simple EMA heuristic if available
        if self.model is None:
            try:
                if "EMA_10" in feat_any.columns and "Close" in feat_any.columns:
                    last = float(feat_any["Close"].iloc[0])
                    ema = float(feat_any["EMA_10"].iloc[0])
                    if last > ema * 1.002:
                        return {"action": "BUY", "score": 0.6}
                    if last < ema * 0.998:
                        return {"action": "SELL", "score": 0.6}
            except Exception as e:
                logger.debug("EMA heuristic failed: %s", e)
            return {"action": "HOLD", "score": 0.0}

        # Use model predict or predict_proba when available
        try:
            if hasattr(self.model, "predict_proba"):
                proba = self.model.predict_proba(Xs)
                # choose positive class probability if 2-class
                score = float(proba[0][1]) if proba.shape[1] > 1 else float(proba[0][0])
                action = "BUY" if score > 0.55 else "HOLD" if score > 0.45 else "SELL"
                return {"action": action, "score": score}

            preds = self.model.predict(Xs)
            v = float(preds[0])
            # interpret numeric prediction: positive -> buy, negative -> sell
            if v > 0.01:
                return {"action": "BUY", "score": min(0.99, float(v))}
            if v < -0.01:
                return {"action": "SELL", "score": min(0.99, float(abs(v)))}
            return {"action": "HOLD", "score": float(abs(v))}
        except Exception as e:
            logger.debug("Model prediction failed: %s", e)
            return {"action": "HOLD", "score": 0.0}

    def scan_symbols(
        self, symbol_ohlcv: Mapping[str, Any], top_n: int = 10
    ) -> Dict[str, Dict[str, Any]]:
        """Pick top_n symbols by recent volume and return predictions."""
        volumes = []
        for s, df in symbol_ohlcv.items():
            try:
                # support pandas DataFrame and list-of-dicts
                if hasattr(df, "columns"):
                    # normalize to lowercase access
                    cols = [c.lower() for c in df.columns]
                    if "volume" in cols:
                        vol = (
                            float(df["volume"].iloc[-1])
                            if "volume" in df.columns
                            else float(df["Volume"].iloc[-1])
                        )
                    else:
                        vol = 0.0
                else:
                    vol = float(df[-1].get("volume", 0.0))
            except Exception:
                vol = 0.0
            volumes.append((s, vol))

        volumes.sort(key=lambda x: x[1], reverse=True)
        selected = [s for s, _ in volumes[:top_n]]
        results: Dict[str, Dict[str, Any]] = {}
        for s in selected:
            try:
                res = self.predict_for_symbol(symbol_ohlcv[s])
            except Exception as e:
                logger.debug("predict_for_symbol failed for %s: %s", s, e)
                res = {"action": "HOLD", "score": 0.0}
            results[s] = res
        return results

    def reload(self) -> None:
        """Reload model/scaler artifacts from disk."""
        self._load()

    def get_metadata(self) -> Optional[dict]:
        base = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
        meta_path = os.path.join(base, "metadata.json")
        if not os.path.exists(meta_path):
            return None
        try:
            import json

            with open(meta_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.debug("Failed to read metadata: %s", e)
            return None

    async def scan_top_by_volume_from_api(
        self, symbols: List[str], top_n: int = 10, limit: int = 240
    ) -> Dict[str, Dict[str, Any]]:
        """Fetch OHLCV for symbols using internal external_data routes and run scan.

        Uses a bounded concurrency semaphore to avoid hammering internal helpers.
        """
        try:
            # Import external_data module directly
            import backend.routes.external_data as external_data
        except Exception as e:
            logger.error("external_data not importable: %s", e)
            raise RuntimeError("external_data endpoint not importable")

        sem = asyncio.Semaphore(6)

        async def _fetch(s: str):
            async with sem:
                try:
                    resp = await external_data.binance_ohlcv(symbol=s, limit=limit)
                    candles = resp.get("candles", [])
                except Exception as e:
                    logger.debug("Failed to fetch candles for %s: %s", s, e)
                    candles = []

                # sentiment/news from internal endpoints; handle errors gracefully
                try:
                    tw = await external_data.twitter_sentiment(symbol=s)
                    sent_score = (
                        tw.get("score", tw.get("sentiment", {}).get("score", 0.0))
                        if isinstance(tw, dict)
                        else 0.0
                    )
                except Exception:
                    logger.debug("twitter_sentiment lookup failed for %s", s)
                    sent_score = 0.0

                # Use CoinGecko trending coins as news proxy
                try:
                    # Import CoinGecko functions
                    from backend.routes.coingecko_data import get_trending_coins
                    trending = await get_trending_coins()
                    # Check if symbol is trending (simplified news proxy)
                    is_trending = any(
                        coin.get("symbol", "").upper() == s.replace("USDT", "").replace("BTC", "").upper()
                        for coin in trending.get("coins", [])[:10]
                    )
                    news_items = [{"trending": True}] if is_trending else []
                except Exception:
                    logger.debug("trending coins lookup failed for %s", s)
                    news_items = []

                # expand sentiment/news into arrays aligned to candles
                n = len(candles)
                sentiment_series = [float(sent_score)] * n
                news_series = [0] * n
                nc = len(news_items)
                if n > 0 and nc > 0:
                    step = max(1, n // nc)
                    placed = 0
                    for i in range(0, n, step):
                        if placed >= nc:
                            break
                        news_series[i] = 1
                        placed += 1

                # attach when list-of-dicts
                if isinstance(candles, list):
                    for idx, row in enumerate(candles):
                        try:
                            row["sentiment"] = sentiment_series[idx]
                            row["news_count"] = news_series[idx]
                        except Exception:
                            logger.debug(
                                "failed to attach sentiment/news to candle idx=%s for %s",
                                idx,
                                s,
                            )

                return s, candles

        tasks = [_fetch(s) for s in symbols]
        results = await asyncio.gather(*tasks)
        symbol_ohlcv = {s: candles for s, candles in results}

        return self.scan_symbols(symbol_ohlcv, top_n=top_n)


def make_default_agent() -> XGBAgent:
    return XGBAgent()
