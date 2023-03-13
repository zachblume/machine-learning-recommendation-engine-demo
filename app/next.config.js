/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  async rewrites() {
    return [
      {
        source: "/api/forwardme",
        destination: "http://35.236.232.41/",
        // permanent: true,
      },
    ];
  },
};

module.exports = nextConfig;
