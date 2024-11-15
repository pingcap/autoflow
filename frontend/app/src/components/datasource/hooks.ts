import { getDatasource } from '@/api/datasources';
import useSWR from 'swr';

export function useDatasource (id: number) {
  const { data: datasource, ...rest } = useSWR(`api.datasource.${id}`, () => getDatasource(id));

  return { datasource, ...rest };
}
