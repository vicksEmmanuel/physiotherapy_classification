import {
  ApolloClient,
  ApolloProvider,
  InMemoryCache,
  createHttpLink,
  from,
  split,
} from '@apollo/client';
import { setContext } from '@apollo/client/link/context';
import { onError } from '@apollo/client/link/error';
import { WebSocketLink } from '@apollo/client/link/ws';
import { getMainDefinition } from '@apollo/client/utilities';
import { SubscriptionClient } from 'subscriptions-transport-ws';
import { Environment } from '../modules/environment';


const wsLink: any = process.browser
  ? new WebSocketLink(
      new SubscriptionClient(`${Environment?.hasuraWebsocket??''}`, {
        reconnect: true,
        timeout: 30000, // Adjust based on your needs
        reconnectionAttempts: 5,
        inactivityTimeout: 0,
        connectionParams: {
          headers: {
            'x-hasura-admin-secret': Environment.hasuraSecret,
          },
        },
      })
    )
  : null;

wsLink?.setOnError((e: any) => {
  console.log(e);
});

const errorLink = onError(({ graphqlError, networkError }: any) => {
  if (graphqlError) {
    graphqlError.map(({ message, locations, path }: any) => {
      return null;
    });
  }
  if (networkError) console.log(`[Network error]: ${networkError}`);
});

const httpLink = createHttpLink({
  uri: Environment.hasuraUrl,
});

const authLink = setContext((_: any, { headers }: any) => {
  // get the authentication token from local storage if it exists
  // const token = localStorage.getItem('token');
  // return the headers to the context so httpLink can read them

  
  return {
    headers: {
      ...headers,
      'x-hasura-admin-secret': Environment.hasuraSecret,
    },
  };
});

const link = from([authLink, errorLink, httpLink]);

const splitLink = process.browser
  ? split(
      ({ query }: any) => {
        const definition = getMainDefinition(query);
        return (
          definition.kind === 'OperationDefinition' &&
          definition.operation === 'subscription'
        );
      },
      wsLink,
      link
    )
  : httpLink;

export const client = new ApolloClient({
  link: splitLink,
  cache: new InMemoryCache(),
});

const Apollo = ({ children }: any) => {
  return <ApolloProvider client={client}>{children}</ApolloProvider>;
};

export default Apollo;
