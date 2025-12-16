/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  async rewrites() {
    // Use backend container name when running in Docker, otherwise localhost
    const backendUrl = process.env.BACKEND_URL || 'http://quantum_backend:8000';
    
    return [
      {
        source: '/api/:path*',
        destination: `${backendUrl}/api/:path*`, // Backend API
      },
    ]
  },
}

module.exports = nextConfig
