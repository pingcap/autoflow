import { CreateEvaluationDatasetItemForm } from '@/components/evaluations/create-evaluation-dataset-item-form';
import type { Meta, StoryObj } from '@storybook/react';

const meta = {
  title: 'Components/Evaluations/CreateEvaluationDatasetItemForm',
  component: CreateEvaluationDatasetItemForm,
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
} satisfies Meta<typeof CreateEvaluationDatasetItemForm>;

export const Default = {
  args: {
    evaluationDatasetId: 8848,
  },
} satisfies StoryObj<typeof meta>;

export default meta;
