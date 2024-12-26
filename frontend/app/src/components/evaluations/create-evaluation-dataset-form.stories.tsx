import { CreateEvaluationDatasetForm } from '@/components/evaluations/create-evaluation-dataset-form';
import type { Meta, StoryObj } from '@storybook/react';

const meta = {
  title: 'Components/Evaluations/CreateEvaluationDatasetForm',
  component: CreateEvaluationDatasetForm,
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
} satisfies Meta<typeof CreateEvaluationDatasetForm>;

export const Default = {} satisfies StoryObj<typeof meta>;

export default meta;
