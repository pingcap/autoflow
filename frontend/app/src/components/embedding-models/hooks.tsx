import { listEmbeddingModels } from '@/api/embedding-models';
import { listAllHelper } from '@/lib/request';
import useSWR from 'swr';

export function useAllEmbeddingModels (flag = true) {
  return useSWR(flag && 'api.embedding-models.list-all', () => listAllHelper(listEmbeddingModels, 'id'));
}

export function useEmbeddingModel (id: number | null | undefined) {
  const { data, mutate, ...rest } = useAllEmbeddingModels(id != null);

  return {
    embeddingModel: data?.find(embeddingModel => embeddingModel.id === id),
    ...rest,
  };
}
