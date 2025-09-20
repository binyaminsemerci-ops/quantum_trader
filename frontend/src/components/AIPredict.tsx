import React, { useEffect, useState } from "react";
import { api as typedApi } from "../utils/api";

const AIPredict: React.FC = () => {
  const [signal, setSignal] = useState<string>("...");

  const refresh = async () => {
    try {
      const res = await typedApi.get<{ signal?: string }>("/predict");
      setSignal(res?.data?.signal ?? "N/A");
    } catch (err: unknown) {
      console.error("AIPredict refresh error", err);
      setSignal("Error");
    }
  };

  useEffect(() => {
    refresh();
  }, []);

  return (
    <div>
      <h2>🤖 AI Prediction</h2>
      <p>
        Signal: <b>{signal}</b>
      </p>
      <button onClick={refresh}>Refresh</button>
    </div>
  );
};

export default AIPredict;
