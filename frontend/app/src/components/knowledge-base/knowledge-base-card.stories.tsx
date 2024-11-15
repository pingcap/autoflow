import { KnowledgeBaseCard } from '@/components/knowledge-base/knowledge-base-card';
import type { Meta, StoryObj } from '@storybook/react';

const meta = {
  title: 'Components/KnowledgeBase/KnowledgeBaseCard',
  component: KnowledgeBaseCard,
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
} satisfies Meta<typeof KnowledgeBaseCard>;

export default meta;

type Story = StoryObj<typeof meta>

export const Default: Story = {
  args: {
    knowledgeBase: {
      id: 1,
      name: 'Some KB',
      description: 'Some Description',
      index_methods: ['vector', 'knowledge_graph'],
      creator: { id: 'xxx' },
      created_at: new Date(),
      updated_at: new Date(),
    },
  },
};
