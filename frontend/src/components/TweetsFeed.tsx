import React, { useEffect, useState } from "react";
import { api as typedApi } from "../utils/api";

type Tweet = { text?: string; sentiment?: string };

const TweetsFeed: React.FC = () => {
  const [tweets, setTweets] = useState<Tweet[]>([]);

  useEffect(() => {
    typedApi.get<Tweet[]>('/tweets?query=bitcoin&limit=3')
      .then((res) => {
        const items = res?.data ?? [];
        setTweets(Array.isArray(items) ? items : []);
      })
      .catch((err: unknown) => {
        console.error('Failed to load tweets', err);
        setTweets([]);
      });
  }, []);

  return (
    <div>
      <h2>🐦 Tweets</h2>
      <ul>
        {tweets.map((t, i) => (
          <li key={i}>{t.text ?? "(no text)"} ({t.sentiment ?? "neutral"})</li>
        ))}
      </ul>
    </div>
  );
};

export default TweetsFeed;
