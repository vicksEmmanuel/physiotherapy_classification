const path = require('path');

const nextConfig = {
  plugins: [
    [
      '@fullhuman/postcss-purgecss',
      {
        content: [
          './pages/**/*.{js,jsx,ts,tsx}',
          './components/**/*.{js,jsx,ts,tsx}',
        ],
        defaultExtractor: (content) => content.match(/[\w-/:]+(?<!:)/g) || [],
      },
    ],
    'postcss-preset-env',
  ],
  reactStrictMode: false,
  swcMinify: true,
  images: {
    dangerouslyAllowSVG: true,
    minimumCacheTTL: 60,
    contentSecurityPolicy: "default-src 'self'; script-src 'none'; sandbox;",
    domains: ['192.168.0.107', 'localhost', 'nomnom-image.s3.amazonaws.com'],
  },
  experimental: {
    serverActions: {
      bodySizeLimit: '1000mb',
    },
  },
  webpack(config) {
    config.module.rules.push({
      test: /\.svg$/,
      use: ['@svgr/webpack'],
    });

    return config;
  },
  async headers() {
    return [
      {
        source: '/:path*',
        headers: [
          {
            key: 'X-XSS-Protection',
            value: '1; mode=block',
          },
          {
            key: 'X-Frame-Options',
            value: 'SAMEORIGIN',
          },
          {
            key: 'X-Content-Type-Options',
            value: 'nosniff',
          },
          {
            key: 'Referrer-Policy',
            value: 'origin-when-cross-origin',
          },
          {
            key: 'Permissions-Policy',
            value: 'geolocation=(self), camera=(), microphone=()',
          },
          // Add this header to disable mixed content protection
          {
            key: 'Content-Security-Policy',
            value: 'block-all-mixed-content',
          },
        ],
      },
    ];
  },
  sassOptions: {
    includePaths: [path.join(__dirname, 'styles')],
  },
  output: 'standalone',
  compress: true,
  matcher: ['/((?!api|_next|.*\\..*).*)'],
};

module.exports = nextConfig;
