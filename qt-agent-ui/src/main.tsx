import React from "react";
import { createRoot } from "react-dom/client";
import "./index.css";
import App from "./App";

const saved = localStorage.getItem("qt-theme");
if (saved) {
  document.documentElement.className = saved === "light" ? "" : `theme-${saved}`;
}

createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
