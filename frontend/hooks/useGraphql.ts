import { Environment } from '@/modules/environment';
import { GraphQLClient } from 'graphql-request';
import * as API from './api/generated';

export function useGraphqlClient() {
  return new GraphQLClient(`${Environment.baseUrl}/api/query`, {
    fetch: (input: RequestInfo, info: RequestInit) => {
      const x = new URL('', input as string);
      return fetch(x as any, info);
    },
  });
}

export function useGraphql() {
  const sdk = API.getSdk(useGraphqlClient(), async (action, name) => {
    console.debug('operation name', name);

    return action();
  });

  return {
    ...sdk,
  };
}
