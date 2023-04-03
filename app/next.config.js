/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  async rewrites() {
    return [
      {
        source: "/api/forwardme",
        destination: "http://35.199.14.111/",
        // permanent: true,
      },
    ];
  },
};

module.exports = nextConfig;
