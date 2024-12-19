import type { EvaluationTask } from '@/api/evaluations';
import { EvaluationTaskInfo, EvaluationTaskInfoDisplay, EvaluationTaskInfoSkeleton } from '@/components/evaluations/evaluation-task-info';
import type { Meta, StoryObj } from '@storybook/react';
import type { ComponentType } from 'react';

const summary = {
  id: 10086,
  dataset_id: 8848,
  name: 'Demo Task',
  created_at: new Date('2024-01-01 18:45:08'),
  updated_at: new Date('2024-01-01 18:45:08'),
  user_id: 'some-user-id',
  summary: {
    errored: 2,
    not_start: 14,
    succeed: 23,
    progressing: 3,
    avg_factual_correctness: 0.83,
    avg_semantic_similarity: 0.76,
    min_factual_correctness: 0.45,
    min_semantic_similarity: 0.56,
    max_factual_correctness: 0.85,
    max_semantic_similarity: 0.95,
    std_factual_correctness: 0.74,
    std_semantic_similarity: 0.83,
  },
} satisfies EvaluationTask;

const meta = {
  title: 'Components/Evaluations/EvaluationTaskSummary',
  subcomponents: {
    EvaluationTaskInfoSkeleton: EvaluationTaskInfoSkeleton,
    EvaluationTaskInfoDisplay: EvaluationTaskInfoDisplay as ComponentType<unknown>,
  },
  args: {},
} satisfies Meta<typeof EvaluationTaskInfo>;

export const Display = {
  render () {
    return (
      <EvaluationTaskInfoDisplay task={summary} />
    );
  },
} satisfies StoryObj<typeof meta>;

export const Skeleton = {
  render () {
    return (
      <EvaluationTaskInfoSkeleton />
    );
  },
} satisfies StoryObj<typeof meta>;

export default meta;