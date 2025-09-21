import React, { useEffect, useState } from "react";
import { api as typedApi } from "../utils/api";

interface NewsItem { title?: string; url?: string; sentiment?: string }

const NewsFeed: React.FC = () => {
  const [news, setNews] = useState<NewsItem[]>([]);

  useEffect(() => {
    typedApi.get<NewsItem[]>('/news?limit=5')
      .then((res) => {
        const items = res?.data ?? [];
        setNews(Array.isArray(items) ? items : []);
      })
      .catch((err: unknown) => {
        console.error('Failed to load news', err);
        setNews([]);
      });
  }, []);

  return (
    <div>
      <h2>📰 News</h2>
      <ul>
        {news.map((n, i) => (
          <li key={i}>
            <a href={n.url ?? '#'} target="_blank" rel="noreferrer">{n.title ?? '(no title)'}</a> ({n.sentiment ?? "neutral"})
          </li>
        ))}
      </ul>
    </div>
  );
};

export default NewsFeed;
