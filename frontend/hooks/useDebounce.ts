import { useEffect, useState } from 'react';

/**
 * Debounces state changes.
 */
export function useDebounce<T>(
  value: T,
  delay: number = 600
): [T, (t: T) => void] {
  const [debouncedValue, setDebouncedValue] = useState(value);

  useEffect(() => {
    const handler = setTimeout(() => {
      setDebouncedValue(value);
    }, delay);

    return () => {
      clearTimeout(handler);
    };
  }, [value, delay]);

  return [debouncedValue, setDebouncedValue];
}
