/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  async rewrites() {
    return [
      {
        source: "/api/forwardme",
        destination: "http://35.245.184.243/",
        // permanent: true,
      },
    ];
  },
};

module.exports = nextConfig;
