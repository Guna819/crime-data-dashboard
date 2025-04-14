import { useCallback } from "react";

const useApiCallLazy = <T>(path: string) => {
  const fetchData = useCallback(async (query?: Record<string, string>) => {
    try {
      const queryString = new URLSearchParams(query).toString();
      const response = await fetch(`http://localhost:8000${path}?${queryString}`);
      if (!response.ok) {
        throw new Error("Error fetching data");
      }
      const result: T = await response.json();
      return result;
    } catch (err: any) {
      throw err;
    }
  }, [path]);

  return { fetchData };
};

export default useApiCallLazy;
