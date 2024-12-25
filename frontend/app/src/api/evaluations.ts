import { authenticationHeaders, handleErrors, handleResponse, type Page, type PageParams, requestUrl, zodPage } from '@/lib/request';
import { zodJson, zodJsonDate } from '@/lib/zod';
import { z, type ZodType } from 'zod';

export interface EvaluationDataset {
  id: number;
  name: string;
  user_id: string;
  created_at: Date;
  updated_at: Date;
}

export interface EvaluationDatasetItem {
  created_at: Date;
  updated_at: Date;
  id: number;
  query: string;
  reference: string;
  retrieved_contexts: string[];
  extra: any;
  evaluation_dataset_id: number;
}

export interface EvaluationTask {
  id: number;
  name: string;
  user_id: string;
  created_at: Date;
  updated_at: Date;
  dataset_id: number;
}

export const EvaluationTaskSummaryMetrics = [
  'avg_factual_correctness',
  'avg_semantic_similarity',
  'min_factual_correctness',
  'min_semantic_similarity',
  'max_factual_correctness',
  'max_semantic_similarity',
  'std_factual_correctness',
  'std_semantic_similarity',
] as const;

export type EvaluationTaskSummaryMetric = typeof EvaluationTaskSummaryMetrics[number];

export interface EvaluationTaskSummary extends Record<EvaluationTaskSummaryMetric, number | null> {
  task: EvaluationTask;
  not_start: number;
  succeed: number;
  errored: number;
  progressing: number;
}

export type EvaluationTaskItemStatus = 'not_start' | 'evaluating' | 'done' | 'error'

export interface EvaluationTaskItem {
  created_at: Date
  updated_at: Date
  id: number
  chat_engine: string
  status: EvaluationTaskItemStatus
  query: string
  reference: string
  response: string | null
  retrieved_contexts: string[] | null
  extra: any | null
  error_msg: string | null
  factual_correctness: number | null
  semantic_similarity: number | null
  evaluation_task_id: number
}

export interface CreateEvaluationDatasetParams {
  name: string;
  upload_id: number;
}

export interface UpdateEvaluationDatasetParams {
  name: string;
}

export interface CreateEvaluationDatasetItemParams {
  query: string;
  reference: string;
  retrieved_contexts: string[];
  extra: any;
}

export interface UpdateEvaluationDatasetItemParams {
  query: string;
  reference: string;
  retrieved_contexts: string[];
  extra: any;
}

export interface CreateEvaluationTaskParams {
  name: string;
  evaluation_dataset_id: number;
  chat_engine?: string;
  run_size?: number;
}

const evaluationDatasetSchema = z.object({
  id: z.number(),
  name: z.string(),
  user_id: z.string(),
  created_at: zodJsonDate(),
  updated_at: zodJsonDate(),
}) satisfies ZodType<EvaluationDataset, any, any>;

const evaluationDatasetItemSchema = z.object({
  created_at: zodJsonDate(),
  updated_at: zodJsonDate(),
  id: z.number(),
  query: z.string(),
  reference: z.string(),
  retrieved_contexts: z.string().array(),
  extra: zodJson(),
  evaluation_dataset_id: z.number(),
}) satisfies ZodType<EvaluationDatasetItem, any, any>;

const evaluationTaskSchema = z.object({
  id: z.number(),
  name: z.string(),
  user_id: z.string(),
  created_at: zodJsonDate(),
  updated_at: zodJsonDate(),
  dataset_id: z.number(),
}) satisfies ZodType<EvaluationTask, any, any>;

const evaluationTaskSummarySchema = z.object({
  task: evaluationTaskSchema,
  not_start: z.number(),
  succeed: z.number(),
  errored: z.number(),
  progressing: z.number(),
  avg_factual_correctness: z.number().nullable(),
  avg_semantic_similarity: z.number().nullable(),
  min_factual_correctness: z.number().nullable(),
  min_semantic_similarity: z.number().nullable(),
  max_factual_correctness: z.number().nullable(),
  max_semantic_similarity: z.number().nullable(),
  std_factual_correctness: z.number().nullable(),
  std_semantic_similarity: z.number().nullable(),
}) satisfies ZodType<EvaluationTaskSummary, any, any>;

const evaluationTaskItemSchema = z.object({
  created_at: zodJsonDate(),
  updated_at: zodJsonDate(),
  id: z.number(),
  chat_engine: z.string(),
  status: z.enum(['not_start', 'evaluating', 'done', 'error']),
  query: z.string(),
  reference: z.string(),
  response: z.string().nullable(),
  retrieved_contexts: z.string().array().nullable(),
  extra: zodJson().nullable(),
  error_msg: z.string().nullable(),
  factual_correctness: z.number().nullable(),
  semantic_similarity: z.number().nullable(),
  evaluation_task_id: z.number(),
}) satisfies ZodType<EvaluationTaskItem, any, any>;

// Datasets

export async function listEvaluationDatasets ({ ...params }: PageParams): Promise<Page<EvaluationDataset>> {
  return fetch(requestUrl('/api/v1/admin/evaluation/datasets', params), {
    headers: await authenticationHeaders(),
  })
    .then(handleResponse(zodPage(evaluationDatasetSchema)));
}

export async function createEvaluationDataset (params: CreateEvaluationDatasetParams): Promise<EvaluationDataset> {
  return fetch(requestUrl('/api/v1/admin/evaluation/datasets'), {
    method: 'POST',
    headers: {
      ...await authenticationHeaders(),
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(params),
  })
    .then(handleResponse(evaluationDatasetSchema));
}

export async function updateEvaluationDataset (id: number, params: UpdateEvaluationDatasetParams): Promise<EvaluationDataset> {
  return fetch(requestUrl(`/api/v1/admin/evaluation/datasets/${id}`), {
    method: 'PUT',
    headers: {
      ...await authenticationHeaders(),
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(params),
  })
    .then(handleResponse(evaluationDatasetSchema));
}

export async function deleteEvaluationDataset (id: number): Promise<void> {
  await fetch(requestUrl(`/api/v1/admin/evaluation/datasets/${id}`), {
    method: 'DELETE',
    headers: {
      ...await authenticationHeaders(),
    },
  });
}

// Dataset Items

export async function listEvaluationDatasetItems (datasetId: number, { ...params }: PageParams): Promise<Page<EvaluationDatasetItem>> {
  return fetch(requestUrl(`/api/v1/admin/evaluation/datasets/${datasetId}/dataset-items`, params), {
    headers: await authenticationHeaders(),
  })
    .then(handleResponse(zodPage(evaluationDatasetItemSchema)));
}

export async function createEvaluationDatasetItem (datasetId: number, params: CreateEvaluationDatasetItemParams): Promise<EvaluationDatasetItem> {
  return fetch(requestUrl(`/api/v1/admin/evaluation/dataset-items`), {
    method: 'POST',
    headers: {
      ...await authenticationHeaders(),
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      evaluation_dataset_id: datasetId,
      ...params,
    }),
  })
    .then(handleResponse(evaluationDatasetItemSchema));
}

export async function updateEvaluationDatasetItem (datasetId: number, id: number, params: UpdateEvaluationDatasetItemParams): Promise<EvaluationDatasetItem> {
  return fetch(requestUrl(`/api/v1/admin/evaluation/dataset-items/${id}`), {
    method: 'PUT',
    headers: {
      ...await authenticationHeaders(),
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      evaluation_dataset_id: datasetId,
      ...params,
    }),
  })
    .then(handleResponse(evaluationDatasetItemSchema));
}

export async function deleteEvaluationDatasetItem (datasetId: number, id: number): Promise<void> {
  await fetch(requestUrl(`/api/v1/admin/evaluation/dataset-items/${id}`), {
    method: 'DELETE',
    headers: {
      ...await authenticationHeaders(),
    },
  })
    .then(handleErrors);
}

// Tasks

export async function createEvaluationTask (params: CreateEvaluationTaskParams): Promise<EvaluationTask> {
  return fetch(requestUrl('/api/v1/admin/evaluation/tasks'), {
    method: 'POST',
    headers: {
      ...await authenticationHeaders(),
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(params),
  })
    .then(handleResponse(evaluationTaskSchema));
}

export async function listEvaluationTasks ({ ...params }: PageParams): Promise<Page<EvaluationTask>> {
  return fetch(requestUrl('/api/v1/admin/evaluation/tasks', params), {
    headers: await authenticationHeaders(),
  })
    .then(handleResponse(zodPage(evaluationTaskSchema)));
}

export async function getEvaluationTaskSummary (id: number): Promise<EvaluationTaskSummary> {
  return fetch(requestUrl(`/api/v1/admin/evaluation/tasks/${id}/summary`), {
    headers: await authenticationHeaders(),
  })
    .then(handleResponse(evaluationTaskSummarySchema));
}

export async function listEvaluationTaskItems (id: number): Promise<EvaluationTaskItem[]> {
  return fetch(requestUrl(`/api/v1/admin/evaluation/tasks/${id}/items`), {
    headers: await authenticationHeaders(),
  })
    .then(handleResponse(evaluationTaskItemSchema.array()));
}
