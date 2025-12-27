import { useEffect, useRef, useState } from "react";
import { WidgetShell, ScreenGrid, span } from "../components/Widget";
import { TrendingUp, TrendingDown, Minus } from "lucide-react";
import { useSignals } from "../hooks/useData";

export default function NavigationScreen() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const { data: signals } = useSignals();
  const [hoveredNode, setHoveredNode] = useState<string | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Set canvas size
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width;
    canvas.height = rect.height;

    // Clear canvas
    ctx.fillStyle = "#0a0a0a";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Draw grid
    ctx.strokeStyle = "rgba(148, 163, 184, 0.1)";
    ctx.lineWidth = 1;
    const gridSize = 50;
    
    for (let x = 0; x < canvas.width; x += gridSize) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, canvas.height);
      ctx.stroke();
    }
    
    for (let y = 0; y < canvas.height; y += gridSize) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(canvas.width, y);
      ctx.stroke();
    }

    // Draw central hub
    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;
    const hubRadius = 40;

    ctx.fillStyle = "#10b981";
    ctx.beginPath();
    ctx.arc(centerX, centerY, hubRadius, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = "white";
    ctx.font = "14px sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText("AI HUB", centerX, centerY);

    // Draw signal nodes around the hub
    const nodes = signals.slice(0, 8);
    const angleStep = (Math.PI * 2) / Math.max(nodes.length, 1);
    const orbitRadius = 180;

    nodes.forEach((signal, i) => {
      const angle = angleStep * i - Math.PI / 2;
      const x = centerX + Math.cos(angle) * orbitRadius;
      const y = centerY + Math.sin(angle) * orbitRadius;
      const nodeRadius = 25;

      // Draw connection line
      ctx.strokeStyle = signal.action === "BUY" 
        ? "rgba(16, 185, 129, 0.3)" 
        : signal.action === "SELL"
        ? "rgba(239, 68, 68, 0.3)"
        : "rgba(148, 163, 184, 0.2)";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(centerX, centerY);
      ctx.lineTo(x, y);
      ctx.stroke();

      // Draw node
      ctx.fillStyle = signal.action === "BUY"
        ? "#10b981"
        : signal.action === "SELL"
        ? "#ef4444"
        : "#64748b";
      ctx.beginPath();
      ctx.arc(x, y, nodeRadius, 0, Math.PI * 2);
      ctx.fill();

      // Draw symbol text
      ctx.fillStyle = "white";
      ctx.font = "10px sans-serif";
      ctx.textAlign = "center";
      ctx.fillText(signal.symbol.slice(0, -4), x, y);
    });

    // Draw legend
    ctx.fillStyle = "rgba(148, 163, 184, 0.7)";
    ctx.font = "12px sans-serif";
    ctx.textAlign = "left";
    ctx.fillText(`${nodes.length} Active Signals`, 20, canvas.height - 20);

  }, [signals]);

  // Top signals for sidebar
  const topSignals = signals.slice(0, 5);

  return (
    <div className="mx-auto max-w-[1280px] px-4 pb-24 pt-4">
      <ScreenGrid>
        <WidgetShell title="Signal Network Map" className={`${span.twoThirds} h-[520px]`}>
          <canvas 
            ref={canvasRef} 
            className="w-full h-full crisp-edges"
          />
        </WidgetShell>

        <WidgetShell title="Active Signals" className={`${span.third} h-[520px]`}>
          <div className="h-full flex flex-col">
            <div className="flex-1 space-y-2 overflow-y-auto pr-1 scrollbar-thin scrollbar-thumb-slate-300 dark:scrollbar-thumb-slate-700 min-h-0">
              {topSignals.map((signal, i) => (
              <div 
                key={i}
                className="p-2.5 rounded-lg bg-black/5 dark:bg-white/5 hover:bg-black/10 dark:hover:bg-white/10 transition-colors cursor-pointer flex-shrink-0"
                onMouseEnter={() => setHoveredNode(signal.symbol)}
                onMouseLeave={() => setHoveredNode(null)}
              >
                <div className="flex items-center justify-between mb-1.5">
                  <div className="flex items-center gap-2">
                    <span className="font-medium text-sm truncate">{signal.symbol}</span>
                    {signal.action === "BUY" ? (
                      <TrendingUp className="w-4 h-4 text-emerald-500 flex-shrink-0" />
                    ) : signal.action === "SELL" ? (
                      <TrendingDown className="w-4 h-4 text-rose-500 flex-shrink-0" />
                    ) : (
                      <Minus className="w-4 h-4 text-slate-400 flex-shrink-0" />
                    )}
                  </div>
                  <span className={`text-xs font-semibold flex-shrink-0 ${
                    signal.action === "BUY" ? "text-emerald-600 dark:text-emerald-400" : 
                    signal.action === "SELL" ? "text-rose-600 dark:text-rose-400" : 
                    "text-slate-400"
                  }`}>
                    {signal.action}
                  </span>
                </div>
                <div className="text-xs text-slate-500 dark:text-slate-400 space-y-0.5">
                  <div className="truncate">Price: ${signal.price?.toFixed(2)}</div>
                  <div className="truncate">Confidence: {(signal.confidence * 100).toFixed(0)}%</div>
                  <div className="text-[10px] text-slate-400 dark:text-slate-500 mt-0.5 truncate">
                    {new Date(signal.timestamp).toLocaleString("no-NO")}
                  </div>
                </div>
              </div>
            ))}
            </div>
          </div>
        </WidgetShell>
      </ScreenGrid>
    </div>
  );
}

