/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  output: 'standalone', // Enable standalone build for Docker
  async rewrites() {
    // Use backend container name when running in Docker, otherwise localhost
    const backendUrl = process.env.BACKEND_URL || 'http://localhost:8000';
    
    return [
      {
        source: '/api/:path*',
        destination: `${backendUrl}/api/:path*`, // Backend API
      },
    ]
  },
}

module.exports = nextConfig
