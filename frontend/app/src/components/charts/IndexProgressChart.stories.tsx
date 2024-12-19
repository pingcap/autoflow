import { IndexProgressChart, IndexProgressChartPlaceholder } from '@/components/charts/IndexProgressChart';
import type { Meta, StoryObj } from '@storybook/react';
import type { FC } from 'react';

const valueType = {
  name: 'number',
  required: false,
} as const;

const meta = {
  title: 'Components/Charts/IndexProgressChart',
  component: IndexProgressChart,
  subcomponents: {
    IndexProgressChartPlaceholder: IndexProgressChartPlaceholder as FC<unknown>,
  },
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
  argTypes: {},
  args: {},
} satisfies Meta<typeof IndexProgressChart>;

export default meta;

type Story = StoryObj<typeof meta>

export const Default: Story = {
  args: {
    title: 'Title',
    description: 'Description',
    label: 'Count',
    data: {
      failed: 23,
      completed: 120,
      not_started: 2,
    },
  },
};

export const Placeholder: StoryObj<typeof IndexProgressChartPlaceholder> = {
  args: {
    title: 'Title',
    description: 'Description',
    label: 'Count',
  },
  render ({ ...args }) {
    return <IndexProgressChartPlaceholder {...args} />;
  },
};
