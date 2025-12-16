import { PropsWithChildren } from "react";
import { cn } from "../lib/cn";

export function WidgetShell({ title, subtitle, children, className }:{
  title?: string; subtitle?: string; children?: any; className?: string;
}) {
  return (
    <section className={cn("card p-4 shadow-card", className)}>
      {(title || subtitle) && (
        <header className="mb-2 flex-shrink-0">
          {title && <h3 className="text-sm font-medium text-slate-600 dark:text-slate-300">{title}</h3>}
          {subtitle && <p className="text-xs text-slate-500">{subtitle}</p>}
        </header>
      )}
      {children}
    </section>
  );
}

/* Grid-container som etterligner wireframe: 12 kolonner, 2–4 rader */
export function ScreenGrid({ children, className }: PropsWithChildren<{className?:string}>) {
  return (
    <div className={cn("grid grid-cols-12 gap-4", className)}>
      {children}
    </div>
  );
}

/* Hjelper for col-span */
export const span = {
  full: "col-span-12",
  half: "col-span-12 md:col-span-6",
  third: "col-span-12 lg:col-span-4",
  twoThirds: "col-span-12 lg:col-span-8",
  quarter: "col-span-12 md:col-span-6 lg:col-span-3",
};
