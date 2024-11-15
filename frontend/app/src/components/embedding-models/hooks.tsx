import { getEmbeddingModelById } from '@/api/embedding-models';
import useSWR from 'swr';

export function useEmbeddingModel (id: number | null | undefined) {
  const { data: embeddingModel, ...rest } = useSWR(id == null ? null : `api.embedding-model.get?id=${id}`, () => getEmbeddingModelById(id as number));
  return { embeddingModel, ...rest };
}
