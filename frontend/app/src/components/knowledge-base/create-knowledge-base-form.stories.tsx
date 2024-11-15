import { CreateKnowledgeBaseForm } from '@/components/knowledge-base/create-knowledge-base-form';
import type { Meta, StoryObj } from '@storybook/react';

const meta = {
  title: 'Components/KnowledgeBase/CreateKnowledgeBaseForm',
  component: CreateKnowledgeBaseForm,
  parameters: {
    layout: 'centered',
  },
  tags: ['autodocs'],
  decorators: [
    (Story) => (
      <div style={{ width: 640 }}>
        <Story />
      </div>
    ),
  ],
  argTypes: {},
  args: {},
} satisfies Meta<typeof CreateKnowledgeBaseForm>;

export default meta;

type Story = StoryObj<typeof meta>

export const Default: Story = {
  args: {
    onCreated () {},
  },
};
