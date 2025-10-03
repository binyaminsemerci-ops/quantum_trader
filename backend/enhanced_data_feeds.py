"""Enhanced Multi-Source Data Integration for AI Learning.

Integrates multiple free APIs to provide comprehensive data feeds:
- CoinGecko: Market data, trends, developer activity
- Fear & Greed Index: Market sentiment indicator
- Reddit Sentiment: Social media analysis
- CryptoCompare: Alternative price feeds & news
- CoinPaprika: Additional market metrics
- Messari: On-chain metrics (free tier)
- Alternative.me: Market indicators
"""

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import aiohttp

logger = logging.getLogger(__name__)


class EnhancedDataFeed:
    """Enhanced data feed aggregator with multiple free APIs."""

    def __init__(self) -> None:
        self.session: Optional[aiohttp.ClientSession] = None
        self.cache: Dict[str, Any] = {}
        self.cache_ttl = 300  # 5 minutes cache

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=10),
            headers={"User-Agent": "Quantum-Trader-AI/1.0"},
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    def _is_cache_valid(self, key: str) -> bool:
        """Check if cached data is still valid."""
        if key not in self.cache:
            return False

        cached_time, _ = self.cache[key]
        return time.time() - cached_time < self.cache_ttl

    def _set_cache(self, key: str, data: Any) -> None:
        """Set data in cache with timestamp."""
        self.cache[key] = (time.time(), data)

    def _get_cache(self, key: str) -> Optional[Any]:
        """Get data from cache if valid."""
        if self._is_cache_valid(key):
            return self.cache[key][1]
        return None

    async def _safe_request(
        self,
        url: str,
        params: Optional[Dict] = None,
    ) -> Optional[Dict]:
        """Make safe HTTP request with error handling."""
        try:
            if not self.session:
                return None

            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.warning(
                        f"API request failed: {url} - Status: {response.status}",
                    )
                    return None

        except asyncio.TimeoutError:
            logger.warning(f"Timeout for API request: {url}")
            return None
        except Exception as e:
            logger.exception(f"Error in API request {url}: {e}")
            return None

    async def get_coingecko_data(self, symbols: List[str]) -> Dict[str, Any]:
        """Get comprehensive CoinGecko market data."""
        cache_key = f"coingecko_{','.join(symbols)}"
        cached = self._get_cache(cache_key)
        if cached:
            return cached

        try:
            # Convert symbols to CoinGecko IDs
            symbol_map = {
                "BTC": "bitcoin",
                "ETH": "ethereum",
                "ADA": "cardano",
                "SOL": "solana",
                "DOT": "polkadot",
                "LINK": "chainlink",
                "UNI": "uniswap",
                "MATIC": "polygon",
            }

            coin_ids = []
            for symbol in symbols:
                clean_symbol = (
                    symbol.replace("USDC", "").replace("USDT", "").replace("USD", "")
                )
                if clean_symbol in symbol_map:
                    coin_ids.append(symbol_map[clean_symbol])

            if not coin_ids:
                return {}

            # Fetch market data
            market_url = "https://api.coingecko.com/api/v3/coins/markets"
            market_params = {
                "ids": ",".join(coin_ids),
                "vs_currency": "usd",
                "order": "market_cap_desc",
                "per_page": 250,
                "page": 1,
                "sparkline": "true",
                "price_change_percentage": "1h,24h,7d,30d",
            }

            market_data = await self._safe_request(market_url, market_params)

            # Fetch global market data for context
            global_url = "https://api.coingecko.com/api/v3/global"
            global_data = await self._safe_request(global_url)

            result = {
                "market_data": market_data or [],
                "global_data": global_data.get("data", {}) if global_data else {},
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "source": "coingecko",
            }

            self._set_cache(cache_key, result)
            logger.info(f"📈 CoinGecko data fetched for {len(coin_ids)} coins")

            return result

        except Exception as e:
            logger.exception(f"CoinGecko API error: {e}")
            return {}

    async def get_fear_greed_index(self) -> Dict[str, Any]:
        """Get Fear & Greed Index from Alternative.me (free)."""
        cache_key = "fear_greed_index"
        cached = self._get_cache(cache_key)
        if cached:
            return cached

        try:
            url = "https://api.alternative.me/fng/"
            params = {"limit": 30}  # Get 30 days of data

            data = await self._safe_request(url, params)

            if data and "data" in data:
                result = {
                    "current": data["data"][0] if data["data"] else {},
                    "historical": data["data"],
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "source": "alternative.me",
                }

                self._set_cache(cache_key, result)
                logger.info(
                    f"😱 Fear & Greed Index: {result['current'].get('value', 'N/A')}",
                )

                return result

            return {}

        except Exception as e:
            logger.exception(f"Fear & Greed Index error: {e}")
            return {}

    async def get_reddit_sentiment(self, symbols: List[str]) -> Dict[str, Any]:
        """Get Reddit sentiment data (using free Reddit API)."""
        cache_key = f"reddit_{','.join(symbols)}"
        cached = self._get_cache(cache_key)
        if cached:
            return cached

        try:
            results: Dict[str, Any] = {}
            for symbol in symbols:
                symbol_sentiment = await self._get_symbol_reddit_sentiment(symbol)
                if symbol_sentiment:
                    results[symbol] = self._calculate_reddit_aggregate(symbol_sentiment)

            result = {
                "symbols": results,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "source": "reddit",
            }

            self._set_cache(cache_key, result)
            logger.info(f"🔴 Reddit sentiment fetched for {len(results)} symbols")
            return result

        except Exception as e:
            logger.exception(f"Reddit sentiment error: {e}")
            return {}

    async def _get_symbol_reddit_sentiment(self, symbol: str) -> List[Dict[str, Any]]:
        """Get Reddit sentiment for a single symbol."""
        clean_symbol = symbol.replace("USDC", "").replace("USDT", "").replace("USD", "")
        subreddits = ["CryptoCurrency", "Bitcoin", "ethereum", "cardano"]
        symbol_sentiment = []

        for subreddit in subreddits:
            try:
                posts = await self._fetch_reddit_posts(subreddit, clean_symbol)
                symbol_sentiment.extend(posts)
                await asyncio.sleep(0.1)  # Be nice to Reddit
            except Exception as e:
                logger.debug(f"Reddit API error for {subreddit}: {e}")
                continue

        return symbol_sentiment

    async def _fetch_reddit_posts(
        self, subreddit: str, symbol: str
    ) -> List[Dict[str, Any]]:
        """Fetch and analyze Reddit posts for a symbol."""
        url = f"https://www.reddit.com/r/{subreddit}/search.json"
        params = {
            "q": symbol,
            "restrict_sr": "true",
            "sort": "new",
            "limit": 10,
        }

        data = await self._safe_request(url, params)
        posts = []

        if data and "data" in data and "children" in data["data"]:
            for post in data["data"]["children"]:
                post_data = post.get("data", {})
                title = post_data.get("title", "")
                score = post_data.get("score", 0)
                comments = post_data.get("num_comments", 0)

                sentiment_score = self._analyze_post_sentiment(title)

                posts.append(
                    {
                        "title": title,
                        "score": score,
                        "comments": comments,
                        "sentiment": sentiment_score,
                        "subreddit": subreddit,
                    }
                )

        return posts

    def _analyze_post_sentiment(self, title: str) -> int:
        """Analyze sentiment of a Reddit post title."""
        positive_words = [
            "moon",
            "bullish",
            "buy",
            "good",
            "great",
            "up",
            "rise",
            "pump",
            "gain",
        ]
        negative_words = [
            "dump",
            "crash",
            "sell",
            "bad",
            "down",
            "fall",
            "bear",
            "drop",
            "loss",
        ]

        sentiment_score = 0
        title_lower = title.lower()

        for word in positive_words:
            if word in title_lower:
                sentiment_score += 1

        for word in negative_words:
            if word in title_lower:
                sentiment_score -= 1

        return sentiment_score

    def _calculate_reddit_aggregate(
        self, symbol_sentiment: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate aggregate sentiment for a symbol."""
        total_sentiment = sum(post["sentiment"] for post in symbol_sentiment)
        avg_sentiment = total_sentiment / len(symbol_sentiment)
        total_engagement = sum(
            post["score"] + post["comments"] for post in symbol_sentiment
        )

        return {
            "sentiment_score": avg_sentiment,
            "total_posts": len(symbol_sentiment),
            "total_engagement": total_engagement,
            "posts": symbol_sentiment[:5],  # Top 5 posts
        }

    async def get_cryptocompare_data(self, symbols: List[str]) -> Dict[str, Any]:
        """Get CryptoCompare data (free tier)."""
        cache_key = f"cryptocompare_{','.join(symbols)}"
        cached = self._get_cache(cache_key)
        if cached:
            return cached

        try:
            results: Dict[str, Any] = {}

            # Get news data
            news_url = "https://min-api.cryptocompare.com/data/v2/news/"
            news_params = {"lang": "EN", "sortOrder": "latest", "lmt": 50}

            news_data = await self._safe_request(news_url, news_params)

            # Get social stats
            for symbol in symbols:
                clean_symbol = (
                    symbol.replace("USDC", "").replace("USDT", "").replace("USD", "")
                )

                social_url = "https://min-api.cryptocompare.com/data/social/coin/latest"
                social_params = {"coinId": clean_symbol}

                social_data = await self._safe_request(social_url, social_params)

                if social_data and "Data" in social_data:
                    results[symbol] = social_data["Data"]

                await asyncio.sleep(0.1)  # Rate limiting

            result = {
                "news": news_data.get("Data", []) if news_data else [],
                "social_stats": results,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "source": "cryptocompare",
            }

            self._set_cache(cache_key, result)
            logger.info(
                f"📰 CryptoCompare data fetched: {len(result['news'])} news items",
            )

            return result

        except Exception as e:
            logger.exception(f"CryptoCompare error: {e}")
            return {}

    async def get_coinpaprika_data(self, symbols: List[str]) -> Dict[str, Any]:
        """Get CoinPaprika market data (free)."""
        cache_key = f"coinpaprika_{','.join(symbols)}"
        cached = self._get_cache(cache_key)
        if cached:
            return cached

        try:
            # Symbol mapping for CoinPaprika
            symbol_map = {
                "BTC": "btc-bitcoin",
                "ETH": "eth-ethereum",
                "ADA": "ada-cardano",
                "SOL": "sol-solana",
                "DOT": "dot-polkadot",
                "LINK": "link-chainlink",
            }

            results: Dict[str, Any] = {}

            for symbol in symbols:
                clean_symbol = (
                    symbol.replace("USDC", "").replace("USDT", "").replace("USD", "")
                )

                if clean_symbol in symbol_map:
                    coin_id = symbol_map[clean_symbol]

                    # Get ticker data
                    ticker_url = f"https://api.coinpaprika.com/v1/tickers/{coin_id}"
                    ticker_data = await self._safe_request(ticker_url)

                    # Get events data
                    events_url = (
                        f"https://api.coinpaprika.com/v1/coins/{coin_id}/events"
                    )
                    events_data = await self._safe_request(events_url)

                    results[symbol] = {
                        "ticker": ticker_data or {},
                        "events": events_data or [],
                    }

                    await asyncio.sleep(0.1)  # Rate limiting

            result = {
                "data": results,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "source": "coinpaprika",
            }

            self._set_cache(cache_key, result)
            logger.info(f"📊 CoinPaprika data fetched for {len(results)} symbols")

            return result

        except Exception as e:
            logger.exception(f"CoinPaprika error: {e}")
            return {}

    async def get_messari_metrics(self, symbols: List[str]) -> Dict[str, Any]:
        """Get Messari on-chain metrics (free tier)."""
        cache_key = f"messari_{','.join(symbols)}"
        cached = self._get_cache(cache_key)
        if cached:
            return cached

        try:
            results = {}

            for symbol in symbols:
                clean_symbol = (
                    symbol.replace("USDC", "").replace("USDT", "").replace("USD", "")
                )

                # Get asset metrics
                metrics_url = f"https://data.messari.io/api/v1/assets/{clean_symbol.lower()}/metrics"
                metrics_data = await self._safe_request(metrics_url)

                if metrics_data and "data" in metrics_data:
                    results[symbol] = metrics_data["data"]

                await asyncio.sleep(0.2)  # Conservative rate limiting

            result = {
                "metrics": results,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "source": "messari",
            }

            self._set_cache(cache_key, result)
            logger.info(f"⛓️ Messari metrics fetched for {len(results)} symbols")

            return result

        except Exception as e:
            logger.exception(f"Messari error: {e}")
            return {}

    async def get_market_indicators(self) -> Dict[str, Any]:
        """Get additional market indicators from Alternative.me."""
        cache_key = "market_indicators"
        cached = self._get_cache(cache_key)
        if cached:
            return cached

        try:
            # Get global crypto stats
            global_url = "https://api.alternative.me/global-crypto-stats"
            global_data = await self._safe_request(global_url)

            result = {
                "global_stats": global_data or {},
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "source": "alternative.me",
            }

            self._set_cache(cache_key, result)
            logger.info("🌍 Global market indicators fetched")

            return result

        except Exception as e:
            logger.exception(f"Market indicators error: {e}")
            return {}

    async def get_all_enhanced_data(self, symbols: List[str]) -> Dict[str, Any]:
        """Get all enhanced data from multiple sources."""
        logger.info(f"🔄 Fetching enhanced data for symbols: {symbols}")

        # Run all API calls concurrently for speed
        tasks = [
            self.get_coingecko_data(symbols),
            self.get_fear_greed_index(),
            self.get_reddit_sentiment(symbols),
            self.get_cryptocompare_data(symbols),
            self.get_coinpaprika_data(symbols),
            self.get_messari_metrics(symbols),
            self.get_market_indicators(),
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        enhanced_data = {
            "coingecko": results[0] if not isinstance(results[0], Exception) else {},
            "fear_greed": results[1] if not isinstance(results[1], Exception) else {},
            "reddit": results[2] if not isinstance(results[2], Exception) else {},
            "cryptocompare": (
                results[3] if not isinstance(results[3], Exception) else {}
            ),
            "coinpaprika": results[4] if not isinstance(results[4], Exception) else {},
            "messari": results[5] if not isinstance(results[5], Exception) else {},
            "indicators": results[6] if not isinstance(results[6], Exception) else {},
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbols_requested": symbols,
        }

        # Extract AI insights from all data
        enhanced_data["ai_insights"] = self._extract_ai_insights(enhanced_data)

        # Log summary
        sources_count = sum(
            1
            for data in enhanced_data.values()
            if isinstance(data, dict) and data.get("source")
        )
        logger.info(f"✅ Enhanced data fetched from {sources_count} sources")

        return enhanced_data

    def _extract_ai_insights(self, enhanced_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract actionable AI insights from enhanced data."""
        insights = {
            "market_sentiment": "neutral",
            "momentum_signals": [],
            "risk_factors": [],
            "opportunity_signals": [],
            "regime_indicators": {},
        }

        try:
            self._process_fear_greed_insights_feeds(insights, enhanced_data)
            self._process_reddit_insights_feeds(insights, enhanced_data)
            self._process_coingecko_insights_feeds(insights, enhanced_data)
            self._process_news_insights_feeds(insights, enhanced_data)
            self._process_global_market_insights_feeds(insights, enhanced_data)

        except Exception as e:
            logger.exception(f"Error extracting AI insights: {e}")
            insights["error"] = str(e)

        return insights

    def _process_fear_greed_insights_feeds(
        self, insights: Dict[str, Any], enhanced_data: Dict[str, Any]
    ) -> None:
        """Process Fear & Greed Index insights."""
        fear_greed = enhanced_data.get("fear_greed", {}).get("current", {})
        if not fear_greed:
            return

        fg_value = fear_greed.get("value", 50)
        if isinstance(fg_value, str):
            try:
                fg_value = int(fg_value)
            except ValueError:
                fg_value = 50

        fg_classification = fear_greed.get("value_classification", "Neutral")
        insights["market_sentiment"] = fg_classification.lower()

        # Extreme fear/greed signals
        if fg_value <= 25:
            insights["opportunity_signals"].append("extreme_fear_buy_opportunity")
        elif fg_value >= 75:
            insights["risk_factors"].append("extreme_greed_sell_signal")

    def _process_reddit_insights_feeds(
        self, insights: Dict[str, Any], enhanced_data: Dict[str, Any]
    ) -> None:
        """Process Reddit sentiment insights."""
        reddit_data = enhanced_data.get("reddit", {}).get("symbols", {})

        for symbol, reddit_info in reddit_data.items():
            sentiment_score = reddit_info.get("sentiment_score", 0)
            total_posts = reddit_info.get("total_posts", 0)

            if total_posts > 5:  # Significant discussion
                if sentiment_score > 0.3:
                    insights["momentum_signals"].append(f"{symbol}_reddit_bullish")
                elif sentiment_score < -0.3:
                    insights["momentum_signals"].append(f"{symbol}_reddit_bearish")

    def _process_coingecko_insights_feeds(
        self, insights: Dict[str, Any], enhanced_data: Dict[str, Any]
    ) -> None:
        """Process CoinGecko market data insights."""
        coingecko_data = enhanced_data.get("coingecko", {})
        market_data = coingecko_data.get("market_data", [])

        for coin in market_data:
            symbol = coin.get("symbol", "").upper()
            price_change_24h = coin.get("price_change_percentage_24h", 0)
            volume_24h = coin.get("total_volume", 0)
            market_cap_rank = coin.get("market_cap_rank", 999)

            # Volume spike detection
            if volume_24h and coin.get("market_cap", 0):
                volume_ratio = volume_24h / coin["market_cap"]
                if volume_ratio > 0.1:  # 10% of market cap in 24h volume
                    insights["momentum_signals"].append(f"{symbol}_volume_spike")

            # Momentum signals from top coins
            if market_cap_rank <= 20:  # Top 20 coins
                if price_change_24h > 10:
                    insights["momentum_signals"].append(f"{symbol}_strong_bullish")
                elif price_change_24h < -10:
                    insights["momentum_signals"].append(f"{symbol}_strong_bearish")

    def _process_news_insights_feeds(
        self, insights: Dict[str, Any], enhanced_data: Dict[str, Any]
    ) -> None:
        """Process news sentiment insights."""
        news_data = enhanced_data.get("cryptocompare", {}).get("news", [])
        positive_news = 0
        negative_news = 0

        positive_keywords = [
            "bullish",
            "rally",
            "surge",
            "adoption",
            "partnership",
            "breakthrough",
        ]
        negative_keywords = ["crash", "dump", "hack", "regulation", "ban", "selloff"]

        for news_item in news_data[:10]:  # Analyze recent news
            title = news_item.get("title", "").lower()

            if any(keyword in title for keyword in positive_keywords):
                positive_news += 1
            elif any(keyword in title for keyword in negative_keywords):
                negative_news += 1

        if positive_news > negative_news and positive_news > 2:
            insights["momentum_signals"].append("positive_news_sentiment")
        elif negative_news > positive_news and negative_news > 2:
            insights["risk_factors"].append("negative_news_sentiment")

    def _process_global_market_insights_feeds(
        self, insights: Dict[str, Any], enhanced_data: Dict[str, Any]
    ) -> None:
        """Process global market indicators."""
        coingecko_data = enhanced_data.get("coingecko", {})
        global_data = coingecko_data.get("global_data", {})

        btc_dominance = 50  # Default fallback
        total_market_cap = 0

        if global_data and isinstance(global_data, dict):
            market_cap_percentages = global_data.get("market_cap_percentage", {})
            if isinstance(market_cap_percentages, dict):
                btc_dominance = market_cap_percentages.get("btc", 50)

            total_mcaps = global_data.get("total_market_cap", {})
            if isinstance(total_mcaps, dict):
                total_market_cap = total_mcaps.get("usd", 0)

        insights["regime_indicators"] = {
            "btc_dominance": btc_dominance,
            "total_market_cap": total_market_cap,
            "market_stage": (
                "altcoin_season" if btc_dominance < 40 else "btc_dominance"
            ),
        }


# Convenience functions for external use
async def get_enhanced_market_data(symbols: List[str]) -> Dict[str, Any]:
    """Get enhanced market data from multiple free sources."""
    async with EnhancedDataFeed() as feed:
        return await feed.get_all_enhanced_data(symbols)


async def get_fear_greed_index() -> Dict[str, Any]:
    """Get current Fear & Greed Index."""
    async with EnhancedDataFeed() as feed:
        return await feed.get_fear_greed_index()


async def get_reddit_crypto_sentiment(symbols: List[str]) -> Dict[str, Any]:
    """Get Reddit sentiment for crypto symbols."""
    async with EnhancedDataFeed() as feed:
        return await feed.get_reddit_sentiment(symbols)


async def get_comprehensive_news() -> Dict[str, Any]:
    """Get comprehensive crypto news from multiple sources."""
    async with EnhancedDataFeed() as feed:
        return await feed.get_cryptocompare_data([])  # News doesn't need symbols
