import { listLlms } from '@/api/llms';
import { listAllHelper } from '@/lib/request';
import useSWR from 'swr';

export function useAllLlms (flag: boolean = true) {
  return useSWR(flag && 'api.llms.list-all', () => listAllHelper(listLlms, 'id'));
}

export function useLlm (id: number | null | undefined) {
  const { data, mutate, ...rest } = useAllLlms(id != null);

  return {
    llm: data?.find(llm => llm.id === id),
    ...rest,
  };
}
