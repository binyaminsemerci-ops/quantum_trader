import { useEffect, useRef } from 'react';

export default function useAutoRefresh(cb: () => void, interval = 5000): void {
  const saved = useRef(cb);
  useEffect(() => {
    saved.current = cb;
  }, [cb]);

  useEffect(() => {
    const id = setInterval(() => saved.current(), interval);
    return () => clearInterval(id);
  }, [interval]);
}
