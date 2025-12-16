import { Battery, Wifi, Clock3 } from "lucide-react";

export default function StatusBar() {
  const now = new Date();
  const hh = now.getHours().toString().padStart(2, "0");
  const mm = now.getMinutes().toString().padStart(2, "0");

  return (
    <div className="status-blur sticky top-0 z-40">
      <div className="mx-auto max-w-[1280px] px-4 h-10 flex items-center justify-between">
        <div className="flex items-center gap-2 text-sm text-slate-500">
          <Wifi size={16} />
          <span>Online</span>
        </div>
        <div className="flex items-center gap-1 text-sm">
          <Clock3 size={16} />
          <span>{hh}:{mm}</span>
        </div>
        <div className="flex items-center gap-2 text-sm text-slate-500">
          <Battery size={16} />
          <span>100%</span>
        </div>
      </div>
    </div>
  );
}
