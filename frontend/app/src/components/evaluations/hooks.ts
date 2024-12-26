import { type EvaluationDataset, type EvaluationTask, listEvaluationDatasets, listEvaluationTasks } from '@/api/evaluations';
import { listAllHelper, ServerError } from '@/lib/request';
import useSWR, { mutate } from 'swr';

export function useAllEvaluationDatasets (flag = true) {
  return useSWR(flag && 'api.evaluation.datasets.list-all', () => listAllHelper(listEvaluationDatasets, 'id'));
}

export function useEvaluationDataset (id: number | null | undefined) {
  const { data, mutate, ...rest } = useAllEvaluationDatasets(id != null);

  let evaluationDataset: EvaluationDataset | undefined;
  let error = rest.error;
  if (data) {
    evaluationDataset = data.find(evaluationDataset => evaluationDataset.id === id);
    if (!evaluationDataset && !error) {
      error = new ServerError(new Response(null, { status: 404 }), 'Not found');
    }
  }

  return {
    evaluationDataset: data?.find(evaluationDataset => evaluationDataset.id === id),
    ...rest,
    error,
  };
}

export function mutateEvaluationDatasets () {
  return mutate(key => {
    if (typeof key === 'string') {
      return key.startsWith(`api.evaluation.datasets.`);
    }
    return false;
  });
}

export function useAllEvaluationTasks (flag = true) {
  return useSWR(flag && 'api.evaluation.tasks.list-all', () => listAllHelper(listEvaluationTasks, 'id'));
}

export function useEvaluationTask (id: number | null | undefined) {
  const { data, mutate, ...rest } = useAllEvaluationTasks(id != null);
  let evaluationTask: EvaluationTask | undefined;
  let error = rest.error;
  if (data) {
    evaluationTask = data.find(evaluationTask => evaluationTask.id === id);
    if (!evaluationTask && !error) {
      error = new ServerError(new Response(null, { status: 404 }), 'Not found');
    }
  }

  return {
    evaluationTask,
    ...rest,
    error,
  };
}

export function mutateEvaluationTasks () {
  return mutate(key => {
    if (typeof key === 'string') {
      return key.startsWith(`api.evaluation.tasks.`);
    }
    return false;
  });
}
