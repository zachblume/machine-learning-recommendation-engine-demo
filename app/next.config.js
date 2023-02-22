/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  async rewrites() {
    return [
      {
        source: "/api/forwardme",
        destination: "http://34.145.135.252/",
        // permanent: true,
      },
    ];
  },
};

module.exports = nextConfig;
