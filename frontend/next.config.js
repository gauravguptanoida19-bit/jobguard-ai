/** @type {import('next').NextConfig} */
const isExport = process.env.OUTPUT_EXPORT === "true" || process.env.GITHUB_ACTIONS === "true";
const basePath = isExport ? "/jobguard-ai" : "";

const nextConfig = {
  reactStrictMode: true,
  ...(isExport
    ? {
        output: "export",
        basePath,
        assetPrefix: `${basePath}/`,
        images: { unoptimized: true },
        trailingSlash: true,
      }
    : {
        output: "standalone",
      }),
};

module.exports = nextConfig;

