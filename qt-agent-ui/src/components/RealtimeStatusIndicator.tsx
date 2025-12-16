import { Wifi, WifiOff } from "lucide-react";
import { useDashboardStream } from "../hooks/useRealtimeData";

export default function RealtimeStatusIndicator() {
  const { isConnected } = useDashboardStream();

  return (
    <div className="fixed top-4 right-4 z-50 flex items-center gap-2 px-3 py-1.5 rounded-lg bg-black/80 dark:bg-white/10 backdrop-blur text-xs">
      {isConnected ? (
        <>
          <Wifi className="w-3 h-3 text-emerald-500" />
          <span className="text-emerald-500">Live</span>
        </>
      ) : (
        <>
          <WifiOff className="w-3 h-3 text-slate-400" />
          <span className="text-slate-400">Connecting...</span>
        </>
      )}
    </div>
  );
}
