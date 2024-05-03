const isProduction = process.env.NEXT_PUBLIC_ENVIRONMENT === 'production';

const productionEnv = {
  baseUrl: isProduction
    ? 'https://physio-video-analyzer.netlify.app'
    : 'http://localhost:3000',
  isProduction: isProduction,
  version: process.env.NEXT_PUBLIC_VERSION,

  hasuraUrl: `${process.env.NEXT_HASURA_PROD_LINK}`,
  hasuraWebsocket: `wss://communal-trout-54.hasura.app/v1/graphql`,
  hasuraSecret: `${process.env.NEXT_HASURA_ADMIN_SECRET}`,

  googleClientSecret: `${process.env.NEXT_GOOGLE_CLIENT_SECRET}`,
  googleClientId: `${process.env.NEXT_GOOGLE_CLIENT_ID}`,

  azureADClientId: `${process.env.NEXT_AZURE_AD_CLIENT_ID}`,
  azureADClientSecret: `${process.env.NEXT_AZURE_AD_CLIENT_SECRET}`,
  azureADTenantId: `${process.env.NEXT_AZURE_AD_TENANT_ID}`,
};

export const Environment = productionEnv;
