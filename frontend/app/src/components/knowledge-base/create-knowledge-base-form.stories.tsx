import { CreateKnowledgeBaseForm } from '@/components/knowledge-base/create-knowledge-base-form';
import type { Meta, StoryObj } from '@storybook/react';
import { AppRouterContext } from 'next/dist/shared/lib/app-router-context.shared-runtime';

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
        <AppRouterContext value={{} as any}>
          <Story />
        </AppRouterContext>
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
