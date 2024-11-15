import { TotalCard } from '@/components/charts/TotalCard';
import type { Meta, StoryObj } from '@storybook/react';
import { GitHubIcon } from 'nextra/icons';

const meta = {
  title: 'Components/Charts/TotalCard',
  component: TotalCard,
  parameters: {
    layout: 'centered',
  },
  tags: ['autodocs'],
  decorators: [
    (Story) => (
      <div style={{ minWidth: 250 }}>
        <Story />
      </div>
    ),
  ],
  argTypes: {
    total: {
      type: 'number',
      control: 'select',
      options: [42, null, undefined],
    },
  },
  args: {},
} satisfies Meta<typeof TotalCard>;

export default meta;

type Story = StoryObj<typeof meta>

export const Default: Story = {
  args: {
    title: 'Title',
    isLoading: false,
    icon: <GitHubIcon />,
    total: 42,
    children: 'Hahaha',
  },
};
