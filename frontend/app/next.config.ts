import MonacoWebpackPlugin from 'monaco-editor-webpack-plugin';
import { NextConfig } from 'next';

const nextConfig: NextConfig = {
  output: process.env.STANDALONE ? 'standalone' : undefined,
  transpilePackages: ['monaco-editor'],
  experimental: {
    optimizePackageImports: ['ai', 'lucide-react'],
    turbo: {
      rules: {
        '*.svg': {
          loaders: ['@svgr/webpack'],
          as: '*.js',
        },
      },
    },
  },
  webpack (config, options) {
    config.module.rules.push({
      test: /\.svg$/,
      use: '@svgr/webpack',
    });
    if (!options.isServer) {
      config.plugins.push(new MonacoWebpackPlugin({
        languages: ['json', 'markdown'],
        filename: 'static/[name].worker.js',
      }));
    }
    return config;
  },
};

export default nextConfig;
