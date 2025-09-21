import React, { useEffect, useState } from "react";
import api from "../api";

const Health: React.FC = () => {
  const [status, setStatus] = useState<string>("...");

  useEffect(() => {
    api.get("/health").then((res: any) => setStatus(res.data?.status ?? "unknown"));
  }, []);

  return <div><b>Backend status:</b> {status}</div>;
};

export default Health;
