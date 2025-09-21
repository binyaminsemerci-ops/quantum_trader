import { useEffect, useState } from 'react';

export default function useDarkMode(): [boolean, (v: boolean) => void] {
  const [enabled, setEnabled] = useState<boolean>(false);

  useEffect(() => {
    if (enabled) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, [enabled]);

  return [enabled, setEnabled];
}
