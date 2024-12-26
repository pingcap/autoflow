import type { EvaluationTaskSummary as EvaluationTaskSummaryType } from '@/api/evaluations';
import { EvaluationTaskSummary, EvaluationTaskSummaryDisplay, EvaluationTaskSummarySkeleton } from '@/components/evaluations/evaluation-task-summary';
import type { Meta, StoryObj } from '@storybook/react';
import type { ComponentType } from 'react';

const summary = {
  task: {
    id: 10086,
    dataset_id: 8848,
    name: 'Demo Task',
    created_at: new Date('2024-01-01 18:45:08'),
    updated_at: new Date('2024-01-01 18:45:08'),
    user_id: 'some-user-id',
  },
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

} satisfies EvaluationTaskSummaryType;

const meta = {
  title: 'Components/Evaluations/EvaluationTaskSummary',
  subcomponents: {
    EvaluationTaskSummarySkeleton: EvaluationTaskSummarySkeleton,
    EvaluationTaskSummaryDisplay: EvaluationTaskSummaryDisplay as ComponentType<unknown>,
  },
  args: {},
} satisfies Meta<typeof EvaluationTaskSummary>;

export const Display = {
  render () {
    return (
      <EvaluationTaskSummaryDisplay summary={summary} />
    );
  },
} satisfies StoryObj<typeof meta>;

export const Skeleton = {
  render () {
    return (
      <EvaluationTaskSummarySkeleton />
    );
  },
} satisfies StoryObj<typeof meta>;

export default meta;