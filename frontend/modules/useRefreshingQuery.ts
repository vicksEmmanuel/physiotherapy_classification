import { useEffect } from 'react';
import { UseQueryState, useQuery } from 'urql';
import { DocumentNode } from 'graphql';

function useRefreshingQuery<T>(
  variables: object,
  query: DocumentNode,
  delayInSec: number,
  pause: boolean
) {
  const [result, reexecuteQuery] = useQuery({
    query: query,
    variables: variables,
    pause: pause,
  });
  useEffect(() => {
    if (result.fetching) {
      return;
    }
    const timerId = setTimeout(() => {
      reexecuteQuery({ requestPolicy: 'network-only' });
    }, delayInSec * 1000);

    return () => clearTimeout(timerId);
  }, [result.fetching, reexecuteQuery, variables]);
  return result as UseQueryState<T, object>;
}

export default useRefreshingQuery;
