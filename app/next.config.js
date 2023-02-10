/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  async rewrites() {
    return [
      {
        source: "/api/forwardme",
        destination: "http://34.86.228.54/",
        // permanent: true,
      },
    ];
  },
};

module.exports = nextConfig;
