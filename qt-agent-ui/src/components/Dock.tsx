import { Home, Bot, BellRing, Navigation, Briefcase, BarChart3 } from "lucide-react";

type ScreenKey = "home" | "trading" | "signals" | "navigation" | "workspace" | "analytics";
export type DockProps = {
  active: ScreenKey;
  onChange: (s: ScreenKey) => void;
  temperature?: number;
  fanMode?: "AUTO" | "LOW" | "HIGH";
};

const Btn = ({ active, label, icon: Icon, onClick }:{
  active:boolean; label:string; icon:any; onClick:()=>void;
}) => (
  <button
    onClick={onClick}
    className={`px-3 py-2 rounded-xl flex items-center gap-2 text-sm ${active ? "bg-brand-600 text-white" : "hover:bg-black/5 dark:hover:bg-white/10"}`}
    aria-label={label}
  >
    <Icon size={18} /> <span className="hidden sm:inline">{label}</span>
  </button>
);

export default function Dock({ active, onChange, temperature=20, fanMode="AUTO" }: DockProps) {
  return (
    <div className="fixed bottom-0 inset-x-0 z-40">
      <div className="mx-auto max-w-[1280px] px-4 py-2 mb-2 rounded-2xl card shadow-card">
        <div className="flex items-center justify-between gap-2">
          <div className="flex items-center gap-2 text-sm">
            <span className="px-2 py-1 rounded-lg bg-black/5 dark:bg-white/10">{temperature}°</span>
            <span className="px-2 py-1 rounded-lg bg-black/5 dark:bg-white/10">{fanMode}</span>
          </div>

          <nav className="flex items-center gap-2">
            <Btn active={active==="home"}      label="Home"      icon={Home}       onClick={()=>onChange("home")} />
            <Btn active={active==="trading"}   label="AI"        icon={Bot}        onClick={()=>onChange("trading")} />
            <Btn active={active==="signals"}   label="Signals"   icon={BellRing}   onClick={()=>onChange("signals")} />
            <Btn active={active==="analytics"} label="Analytics" icon={BarChart3}  onClick={()=>onChange("analytics")} />
            <Btn active={active==="navigation"}label="Nav"       icon={Navigation} onClick={()=>onChange("navigation")} />
            <Btn active={active==="workspace"} label="Work"      icon={Briefcase}  onClick={()=>onChange("workspace")} />
          </nav>

          <div className="flex items-center gap-2 text-sm">
            <span className="px-2 py-1 rounded-lg bg-black/5 dark:bg-white/10">× 2</span>
          </div>
        </div>
      </div>
    </div>
  );
}
