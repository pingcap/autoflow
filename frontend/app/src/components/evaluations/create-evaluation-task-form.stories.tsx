import { CreateEvaluationTaskForm } from '@/components/evaluations/create-evaluation-task-form';
import type { Meta, StoryObj } from '@storybook/react';

const meta = {
  title: 'Components/Evaluations/CreateEvaluationTaskForm',
  component: CreateEvaluationTaskForm,
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
} satisfies Meta<typeof CreateEvaluationTaskForm>;

export const Default = {} satisfies StoryObj<typeof meta>;

export default meta;
