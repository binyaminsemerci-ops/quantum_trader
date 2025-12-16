import { useState } from "react";
import StatusBar from "./components/StatusBar";
import Dock from "./components/Dock";
import HomeScreen from "./screens/HomeScreen";
import TradingScreen from "./screens/TradingScreen";
import SignalsScreen from "./screens/SignalsScreen";
import NavigationScreen from "./screens/NavigationScreen";
import WorkspaceScreen from "./screens/WorkspaceScreen";
import AnalyticsScreen from "./screens/AnalyticsScreen";

type ScreenKey = "home" | "trading" | "signals" | "navigation" | "workspace" | "analytics";

function ThemeSwitch() {
  const [theme, setTheme] = useState<"light"|"dark"|"blue">(
    (localStorage.getItem("qt-theme") as any) || "light"
  );
  const apply = (t:"light"|"dark"|"blue") => {
    setTheme(t);
    localStorage.setItem("qt-theme", t);
    document.documentElement.className = t === "light" ? "" : `theme-${t}`;
  };
  return (
    <div className="flex items-center gap-2 text-sm">
      <span className="text-slate-500">Theme</span>
      <button onClick={()=>apply("light")} className={`px-2 py-1 rounded-lg ${theme==="light"?"bg-black/5 dark:bg-white/10":""}`}>Light</button>
      <button onClick={()=>apply("dark")} className={`px-2 py-1 rounded-lg ${theme==="dark"?"bg-black/5 dark:bg-white/10":""}`}>Dark</button>
      <button onClick={()=>apply("blue")} className={`px-2 py-1 rounded-lg ${theme==="blue"?"bg-black/5 dark:bg-white/10":""}`}>Blue</button>
    </div>
  );
}

export default function App() {
  const [screen, setScreen] = useState<ScreenKey>("home");

  return (
    <div className="min-h-dvh">
      <StatusBar />
      {/* Tittel- og filterlinje */}
      <div className="mx-auto max-w-[1280px] px-4 py-3 flex items-center justify-between">
        <div className="flex items-baseline gap-3">
          <h1 className="text-xl font-semibold">Quantum Trader</h1>
          <div className="hidden md:flex items-center gap-2 text-sm">
            <select className="px-2 py-1 rounded-lg bg-black/5 dark:bg-white/10">
              <option>BTCUSDT</option>
              <option>ETHUSDT</option>
              <option>SOLUSDT</option>
            </select>
            <select className="px-2 py-1 rounded-lg bg-black/5 dark:bg-white/10">
              <option>1m</option>
              <option>5m</option>
              <option>1h</option>
              <option>1d</option>
            </select>
            <button className="px-2 py-1 rounded-lg bg-black/5 dark:bg-white/10">24h</button>
            <button className="px-2 py-1 rounded-lg bg-black/5 dark:bg-white/10">7d</button>
            <button className="px-2 py-1 rounded-lg bg-black/5 dark:bg-white/10">30d</button>
          </div>
        </div>
        <ThemeSwitch />
      </div>

      {/* Skjerm-bytter (wireframe-layout i egne komponenter) */}
      {screen === "home" && <HomeScreen />}
      {screen === "trading" && <TradingScreen />}
      {screen === "signals" && <SignalsScreen />}
      {screen === "analytics" && <AnalyticsScreen />}
      {screen === "navigation" && <NavigationScreen />}
      {screen === "workspace" && <WorkspaceScreen />}

      {/* Dock-navigasjon (alltid synlig, som i wireframe) */}
      <Dock active={screen} onChange={setScreen} />
    </div>
  );
}
