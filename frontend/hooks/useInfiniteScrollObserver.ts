import { useRouter } from "next/router";
import { useEffect } from "react";

/**
 * Handles when the view port is close to the reference, an action param is provided so as to perform and action when a user scrolls close to the viewport of the reference
 *
 * @param reference
 * @param onAction
 * @param condition
 */
const useInfiniteScrollObserver = ({
  condition,
  reference,
  onAction,
}: {
  condition: boolean;
  reference: any;
  onAction?(): void;
}) => {
  const router = useRouter();
  useEffect(() => {
    let intersectionObserver: any;
    if (condition) {
      intersectionObserver = new IntersectionObserver(
        (e) => {
          if (e[0].intersectionRatio <= 0) return;

          onAction?.();
        },
        {
          root: null,
          rootMargin: "0px",
          threshold: [0.0, 0.75, 1],
        },
      );

      try {
        intersectionObserver?.observe?.(reference);
      } catch {}
    } else {
      try {
        intersectionObserver?.disconnect?.();
      } catch {}
    }
  }, [condition, router, reference]);
};

export default useInfiniteScrollObserver;
