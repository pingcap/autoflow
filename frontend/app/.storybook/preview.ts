import type { Preview } from "@storybook/react";
import '../src/app/globals.css';
import '../src/app/chart-theme.css';

const preview: Preview = {
  parameters: {
    controls: {
      matchers: {
        color: /(background|color)$/i,
        date: /Date$/i,
      },
    },
  },
};

export default preview;
