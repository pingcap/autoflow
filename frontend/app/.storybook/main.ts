import type { StorybookConfig } from '@storybook/nextjs';
import MonacoEditorWebpackPlugin = require('monaco-editor-webpack-plugin');

const config: StorybookConfig = {
  stories: [
    '../src/!(pages)/**/*.stories.@(js|jsx|mjs|ts|tsx|mdx)',
  ],
  addons: [
    '@storybook/addon-onboarding',
    '@storybook/addon-links',
    '@storybook/addon-essentials',
    '@chromatic-com/storybook',
    '@storybook/addon-interactions',
  ],
  framework: {
    name: '@storybook/nextjs',
    options: {},
  },
  webpack (config) {
    config.plugins.push(new MonacoEditorWebpackPlugin({
      languages: ['json'],
      filename: 'static/[name].worker.js',
    }));
    return config;
  },
  staticDirs: ['../public'],
};
export default config;
