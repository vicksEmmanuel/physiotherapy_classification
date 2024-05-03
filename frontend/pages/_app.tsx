'use client';
import { CacheProvider } from '@chakra-ui/next-js';
import { ChakraProvider, extendTheme } from '@chakra-ui/react';
import dynamic from 'next/dynamic';
import Head from 'next/head';
import 'react-draft-wysiwyg/dist/react-draft-wysiwyg.css';
import 'react-phone-input-2/lib/style.css';

import Apollo from '@/components/Apollo';
import HeaderBlock from '@/components/HeaderBlock';
import { MultiSelectTheme } from 'chakra-multiselect';
import { SessionProvider } from 'next-auth/react';
import 'react-chat-elements/dist/main.css';
import 'react-loading-skeleton/dist/skeleton.css';
import { QueryClient, QueryClientProvider } from 'react-query';
import { ToastContainer } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
import '../styles/globals.css';

const queryClient = new QueryClient();

const theme = extendTheme({
  colors: {
    green: {
      100: '#00B074',
      200: '#00B074',
      300: '#00B074',
      400: '#00B074',
      500: '#00B074',
      600: '#00B074',
      700: '#00B074',
      800: '#00B074',
      900: '#00B074',
    },
  },
  components: {
    MultiSelect: MultiSelectTheme,
  },
});

const ErrorBoundary = dynamic(
  () => import('../components/ErrorBoundary')
) as any;

function MyApp({ Component, pageProps, session }: any & { session: any }) {

  console.log("About to start the app")

  return (
    <SessionProvider session={session}>
      <CacheProvider>
        <QueryClientProvider client={queryClient}>
          <Head>
           <HeaderBlock />
          </Head>
          <ErrorBoundary>
            <ChakraProvider theme={theme}>
              <Apollo>
                  <Component {...pageProps} />
              </Apollo>
            </ChakraProvider>
          </ErrorBoundary>
          <ToastContainer />
        </QueryClientProvider>
      </CacheProvider>
    </SessionProvider>
  );
}


export default MyApp;
