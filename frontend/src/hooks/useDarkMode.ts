import { useEffect, useState } from 'react';
import type { Dispatch, SetStateAction } from 'react';

export default function useDarkMode(): [boolean, Dispatch<SetStateAction<boolean>>] {
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
