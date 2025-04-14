import { useState, useEffect } from 'react';

type ApiResponse<T> = {
  data: T | null;
  error: string | null;
  loading: boolean;
};

const useApiCall = <T>(path: string): ApiResponse<T> => {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await fetch(`http://localhost:8000${path}`);
        if (!response.ok) {
          throw new Error('Error fetching data');
        }
        const result: T = await response.json();
        setData(result);
      } catch (err: any) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, [path]);

  return { data, error, loading };
};

export default useApiCall;
