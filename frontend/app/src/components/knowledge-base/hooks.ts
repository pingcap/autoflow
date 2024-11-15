import { getKnowledgeGraphIndexProgress } from '@/api/knowledge-base';
import { useKB } from '@/app/(main)/(admin)/knowledge-bases/[id]/context';
import useSWR from 'swr';

export function useKnowledgeBaseDatasource (id: number) {
  const { data_sources } = useKB();

  return data_sources.find(ds => ds.id === id);
}


export function useKnowledgeBaseIndexProgress (id: number) {
  const { data: progress, ...rest  } = useSWR(`api.knowledge-base.${id}.index-progress`, () => getKnowledgeGraphIndexProgress(id));
  return { progress, ...rest };
}
