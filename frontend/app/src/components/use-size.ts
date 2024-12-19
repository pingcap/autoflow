import { useLayoutEffect, useRef, useState } from 'react';

export function useSize () {
  const [size, setSize] = useState<DOMRectReadOnly | undefined>(undefined);
  const ref = useRef<HTMLDivElement>(null);
  useLayoutEffect(() => {
    const el = ref.current;
    if (el) {
      const ro = new ResizeObserver(() => {
        setSize(el.getBoundingClientRect());
      });

      setSize(el.getBoundingClientRect());
      ro.observe(el);
      ro.observe(document.documentElement);

      // observe sidebar inset
      for (let main of Array.from(document.getElementsByTagName('main'))) {
        ro.observe(main);
      }

      return () => {
        ro.disconnect();
      };
    }
  }, []);

  return {
    ref, size,
  };
}