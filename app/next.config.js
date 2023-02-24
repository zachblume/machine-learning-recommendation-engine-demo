/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  async rewrites() {
    return [
      {
        source: "/api/forwardme",
        destination: "http://34.145.243.68/",
        // permanent: true,
      },
    ];
  },
};

module.exports = nextConfig;
