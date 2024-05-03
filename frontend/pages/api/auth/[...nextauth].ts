import { Environment } from '@/modules/environment';
import NextAuth from 'next-auth';
import AzureADProvider from 'next-auth/providers/azure-ad';
import GoogleProvider from 'next-auth/providers/google';

export const authOptions = {
  providers: [
    GoogleProvider({
      clientId: Environment.googleClientId,
      clientSecret: Environment.googleClientSecret,
    }),
    AzureADProvider({
      clientId: Environment.azureADClientId,
      clientSecret: Environment.azureADClientSecret,
      // tenantId: Environment.azureADTenantId,
      tenantId: 'common',
      authorization: {
        params: {
          scope: 'openid profile User.Read email',
        },
        url: `https://login.microsoftonline.com/common/oauth2/v2.0/authorize`,
      },
      profile: (profile: any) => {
        return {
          id: profile.oid,
          name: profile?.name ?? profile?.preferred_username,
          email: profile?.preferred_username ?? profile?.email,
          image: '',
        };
      },
    }),
  ],
  secret: 'RockmyBoat',
  pages: {
    error: '/', // The root directory is usually mapped to the homepage
  },
};

export default NextAuth({ ...authOptions });
