/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  env: {
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL || 'https://api.quantumfond.com',
    NEXT_PUBLIC_AUTH_URL: process.env.NEXT_PUBLIC_AUTH_URL || 'https://auth.quantumfond.com',
  },
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: 'https://api.quantumfond.com/:path*',
      },
    ];
  },
};

module.exports = nextConfig;
