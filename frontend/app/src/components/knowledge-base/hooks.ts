import { listDataSources } from '@/api/datasources';
import { getKnowledgeGraphIndexProgress, listKnowledgeBases } from '@/api/knowledge-base';
import { listAllHelper } from '@/lib/request';
import useSWR, { mutate } from 'swr';

export function useKnowledgeBaseIndexProgress (id: number) {
  const { data: progress, ...rest } = useSWR(`api.knowledge-base.${id}.index-progress`, () => getKnowledgeGraphIndexProgress(id));
  return { progress, ...rest };
}

export function useAllKnowledgeBases (flag = true) {
  return useSWR(flag && `api.knowledge-bases.list-all`, () => listAllHelper(listKnowledgeBases, 'id'));
}

export function useKnowledgeBase (id: number | null | undefined) {
  const { data, mutate, ...rest } = useAllKnowledgeBases(id != null);

  return {
    knowledgeBase: data?.find(llm => llm.id === id),
    ...rest,
  };
}

export function useAllKnowledgeBaseDataSources (kbId: number, flag = true) {
  return useSWR(flag && `api.knowledge-bases.${kbId}.data-sources.list-all`, () => listAllHelper((params) => listDataSources(kbId, params), 'id'));
}

export function mutateKnowledgeBases () {
  return mutate(key => {
    if (typeof key === 'string') {
      return key.startsWith(`api.knowledge-bases.`);
    }
    return false;
  });
}