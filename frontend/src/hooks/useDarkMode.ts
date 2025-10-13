import { useEffect, useState } from 'react';
import type { Dispatch, SetStateAction } from 'react';

export default function useDarkMode(): [boolean, Dispatch<SetStateAction<boolean>>] {
  const [enabled, setEnabled] = useState<boolean>(() => {
    const stored = localStorage.getItem('qt_dark');
    if (stored === '1') return true;
    if (stored === '0') return false;
    // fallback: prefer system
    return window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
  });

  useEffect(() => {
    const root = document.documentElement;
    const body = document.body;
    if (enabled) {
      root.classList.add('dark');
      body.classList.add('dark');
      localStorage.setItem('qt_dark', '1');
    } else {
      root.classList.remove('dark');
      body.classList.remove('dark');
      localStorage.setItem('qt_dark', '0');
    }
  }, [enabled]);

  return [enabled, setEnabled];
}
