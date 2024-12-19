import { listRerankers } from '@/api/rerankers';
import { listAllHelper } from '@/lib/request';
import useSWR from 'swr';

export function useAllRerankers (flag = true) {
  return useSWR(flag && 'api.rerankers.list-all', () => listAllHelper(listRerankers, 'id'));
}

export function useReranker (id: number | null | undefined) {
  const { data, mutate, ...rest } = useAllRerankers(id != null);

  return {
    reranker: data?.find(reranker => reranker.id === id),
    ...rest,
  };
}
