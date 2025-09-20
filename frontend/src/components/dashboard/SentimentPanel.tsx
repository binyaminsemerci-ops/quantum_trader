// Consolidated, typed SentimentPanel
import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Pie } from 'react-chartjs-2';
import { Chart as ChartJS, ArcElement, Tooltip, Legend } from 'chart.js';
import moment from 'moment';

ChartJS.register(ArcElement, Tooltip, Legend);

type SentimentSummary = {
  summary?: { positive?: number; neutral?: number; negative?: number };
  positive_count?: number;
  neutral_count?: number;
  negative_count?: number;
};

type NewsItem = { id: string | number; published_at?: string; url?: string; title?: string; source?: string; sentiment?: string };
type TweetItem = { id: string | number; created_at?: string; text?: string; sentiment_score?: number };

const SentimentPanel: React.FC<{ symbol?: string }> = ({ symbol = 'BTC' }) => {
  const [newsSentiment, setNewsSentiment] = useState<SentimentSummary | null>(null);
  const [tweetSentiment, setTweetSentiment] = useState<SentimentSummary | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [recentNews, setRecentNews] = useState<NewsItem[]>([]);
  const [recentTweets, setRecentTweets] = useState<TweetItem[]>([]);

  useEffect(() => {
    let mounted = true;
    const fetchSentimentData = async () => {
      setLoading(true);
      try {
        const newsResponse = await axios.get<{ summary?: { positive?: number; neutral?: number; negative?: number }; positive_count?: number; neutral_count?: number; negative_count?: number }>('http://localhost:8000/news/sentiment/summary?days=7');
        const tweetResponse = await axios.get<SentimentSummary>(`http://localhost:8000/tweets/sentiment/summary?hours=24&symbol=${symbol}`);
        const recentNewsResponse = await axios.get<NewsItem[]>(`http://localhost:8000/news?limit=5&currency=${symbol}`);
        const recentTweetsResponse = await axios.get<TweetItem[]>(`http://localhost:8000/tweets?limit=5&symbol=${symbol}`);

        if (!mounted) return;
        setNewsSentiment(newsResponse.data ?? null);
        setTweetSentiment(tweetResponse.data ?? null);
        setRecentNews(Array.isArray(recentNewsResponse.data) ? recentNewsResponse.data : []);
        setRecentTweets(Array.isArray(recentTweetsResponse.data) ? recentTweetsResponse.data : []);
      } catch (err) {
        console.error('Error fetching sentiment data:', err);
        if (!mounted) return;
        setError('Failed to load sentiment data. Please try again later.');
      } finally {
        if (mounted) setLoading(false);
      }
    };

    fetchSentimentData();
    const timerInterval = setInterval(fetchSentimentData, 15 * 60 * 1000);
    return () => {
      mounted = false;
      clearInterval(timerInterval);
    };
  }, [symbol]);

  const newsSentimentChartData = {
    labels: ['Positive', 'Neutral', 'Negative'],
    datasets: [
      {
        data: newsSentiment ? [
          newsSentiment.summary?.positive ?? 0,
          newsSentiment.summary?.neutral ?? 0,
          newsSentiment.summary?.negative ?? 0
        ] : [0, 0, 0],
        backgroundColor: ['rgba(75, 192, 192, 0.6)', 'rgba(255, 206, 86, 0.6)', 'rgba(255, 99, 132, 0.6)'],
        borderColor: ['rgba(75, 192, 192, 1)', 'rgba(255, 206, 86, 1)', 'rgba(255, 99, 132, 1)'],
        borderWidth: 1
      }
    ]
  };

  const tweetSentimentChartData = {
    labels: ['Positive', 'Neutral', 'Negative'],
    datasets: [
      {
        data: tweetSentiment ? [
          (tweetSentiment.summary?.positive ?? tweetSentiment.positive_count) ?? 0,
          (tweetSentiment.summary?.neutral ?? tweetSentiment.neutral_count) ?? 0,
          (tweetSentiment.summary?.negative ?? tweetSentiment.negative_count) ?? 0
        ] : [0, 0, 0],
        backgroundColor: ['rgba(75, 192, 192, 0.6)', 'rgba(255, 206, 86, 0.6)', 'rgba(255, 99, 132, 0.6)'],
        borderColor: ['rgba(75, 192, 192, 1)', 'rgba(255, 206, 86, 1)', 'rgba(255, 99, 132, 1)'],
        borderWidth: 1
      }
    ]
  };

  const chartOptions: Record<string, unknown> = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'bottom' as const
      }
    }
  };

  if (loading) return <div className="sentiment-panel loading">Loading sentiment data...</div>;
  if (error) return <div className="sentiment-panel error">{error}</div>;

  return (
    <div className="sentiment-panel">
      <h3>Market Sentiment Analysis</h3>
      <div className="sentiment-charts">
        <div className="chart-container">
          <h4>News Sentiment (7 days)</h4>
          <div style={{ height: '200px' }}>
            <Pie data={newsSentimentChartData} options={chartOptions} />
          </div>
        </div>
        <div className="chart-container">
          <h4>Twitter Sentiment (24 hours)</h4>
          <div style={{ height: '200px' }}>
            <Pie data={tweetSentimentChartData} options={chartOptions} />
          </div>
        </div>
      </div>
      <div className="sentiment-data-sources">
        <div className="news-preview">
          <h4>Recent News</h4>
          <ul className="news-list">
            {recentNews.length > 0 ? (
              recentNews.map(news => (
                <li key={news.id} className={`news-item ${news.sentiment}`}>
                  <span className="news-date">{moment(news.published_at).format('MM/DD HH:mm')}</span>
                  <a href={news.url} target="_blank" rel="noopener noreferrer">{news.title}</a>
                  <span className="news-source">{news.source}</span>
                </li>
              ))
            ) : (
              <li>No recent news found</li>
            )}
          </ul>
        </div>
        <div className="tweets-preview">
          <h4>Recent Tweets</h4>
          <ul className="tweets-list">
            {recentTweets.length > 0 ? (
              recentTweets.map(tweet => (
                <li key={tweet.id} className="tweet-item">
                  <span className="tweet-date">{moment(tweet.created_at).format('MM/DD HH:mm')}</span>
                  <span className="tweet-text">{(tweet.text ?? '').substring(0, 100)}...</span>
                  <span className={`tweet-sentiment ${(tweet.sentiment_score ?? 0) > 0 ? 'positive' : (tweet.sentiment_score ?? 0) < 0 ? 'negative' : 'neutral'}`}>
                    {(tweet.sentiment_score ?? 0) > 0.2 ? '😀' : (tweet.sentiment_score ?? 0) < -0.2 ? '😞' : '😐'}
                    {(tweet.sentiment_score ?? 0).toFixed(2)}
                  </span>
                </li>
              ))
            ) : (
              <li>No recent tweets found</li>
            )}
          </ul>
        </div>
      </div>
    </div>
  );
};

export default SentimentPanel;
