// frontend/src/hooks/useDarkMode.jsx
import { useEffect, useState } from "react";

export default function useDarkMode() {
  const [enabled, setEnabled] = useState(false);

  useEffect(() => {
    if (enabled) {
      document.documentElement.classList.add("dark");
    } else {
      document.documentElement.classList.remove("dark");
    }
  }, [enabled]);

  return [enabled, setEnabled];
}
