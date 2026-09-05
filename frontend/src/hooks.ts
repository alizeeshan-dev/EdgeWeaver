import { useCallback, useEffect, useState } from "react";

export function useRemote<T>(loader: () => Promise<T>, dependencies: React.DependencyList = []) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const reload = useCallback(() => {
    setLoading(true);
    setError(null);
    void loader()
      .then(setData)
      .catch((reason: unknown) => setError(reason instanceof Error ? reason.message : String(reason)))
      .finally(() => setLoading(false));
    // The caller controls when a loader should be re-evaluated.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, dependencies);

  useEffect(reload, [reload]);
  return { data, error, loading, reload };
}
