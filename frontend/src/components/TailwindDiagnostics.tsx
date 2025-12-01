import { useEffect, useState } from 'react';

interface ResultState {
  darkClassActive: boolean;
  darkVariantWorks: boolean | null; // null = pending
  probeLight: string;
  probeDark: string;
}

export default function TailwindDiagnostics() {
  const [open, setOpen] = useState(false);
  const [res, setRes] = useState<ResultState>({ darkClassActive: document.documentElement.classList.contains('dark'), darkVariantWorks: null, probeLight: '', probeDark: '' });

  useEffect(() => {
    // create a probe element
    const probe = document.createElement('div');
    probe.className = 'bg-gray-100 dark:bg-gray-900 h-4 w-4 fixed -left-[9999px] -top-[9999px]';
    document.body.appendChild(probe);

    const html = document.documentElement;
    const hadDark = html.classList.contains('dark');

    // Ensure starting without dark for light measurement
    html.classList.remove('dark');
    const light = getComputedStyle(probe).backgroundColor;

    // Add dark and force reflow
    html.classList.add('dark');
    const dark = getComputedStyle(probe).backgroundColor;

    // Restore original state
    if (!hadDark) html.classList.remove('dark');

    const works = light !== dark; // If colors differ, dark variant applied
    setRes({ darkClassActive: hadDark, darkVariantWorks: works, probeLight: light, probeDark: dark });

    return () => { probe.remove(); };
  }, []);

  return (
    <div style={{ position: 'fixed', bottom: 10, left: 10, zIndex: 9999 }}>
      <button onClick={() => setOpen(o => !o)} className="px-2 py-1 text-xs rounded bg-black/60 text-white hover:bg-black/80 transition">
        {open ? 'Close Diagnostics' : 'Diagnostics'}
      </button>
      {open && (
        <div className="mt-2 p-3 rounded bg-black/75 text-[11px] text-white space-y-1 font-mono w-72 backdrop-blur">
          <div className="font-semibold">Tailwind Diagnostics</div>
          <div>html has dark class: {String(res.darkClassActive)}</div>
          <div>dark variant works: {res.darkVariantWorks === null ? 'checking...' : String(res.darkVariantWorks)}</div>
          <div>probe light bg: {res.probeLight}</div>
            <div>probe dark bg: {res.probeDark}</div>
          {res.darkVariantWorks === false && (
            <div className="text-red-300 mt-1">Dark variant NOT applying. Sjekk tailwind.config content paths.</div>
          )}
          {res.darkVariantWorks && (
            <div className="text-green-300 mt-1">Dark variant OK ✅</div>
          )}
          <div className="pt-1 border-t border-white/10">
            <button onClick={() => { document.documentElement.classList.toggle('dark'); setRes(s => ({ ...s, darkClassActive: document.documentElement.classList.contains('dark') })); }} className="mr-2 underline">toggle dark</button>
            <button onClick={() => { document.documentElement.classList.toggle('compact-mode'); }} className="underline">toggle compact</button>
          </div>
        </div>
      )}
    </div>
  );
}
