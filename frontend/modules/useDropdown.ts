import { useEffect, useRef, useState } from 'react';

export const useDropdown = <T extends HTMLElement = HTMLDivElement>() => {
  const ref = useRef<T>(null);
  const [showMenu, setShowMenu] = useState(false);
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        setShowMenu(false);
      }
    };

    window.addEventListener('click', handler, true);
    return () => {
      window.removeEventListener('click', handler, true);
    };
  }, []);
  return { ref, showMenu, setShowMenu };
};
