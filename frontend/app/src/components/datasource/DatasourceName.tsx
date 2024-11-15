'use client';

import { getDatasource } from '@/api/datasources';
import { cn } from '@/lib/utils';
import { Loader2Icon } from 'lucide-react';
import useSWR from 'swr';

export function DatasourceName ({ id }: { id: number }) {
  const { data: datasource, isLoading, isValidating } = useSWR(`api.datasources.${id}`, () => getDatasource(id));

  if (isLoading) {
    return <Loader2Icon className="size-4 animate-spin repeat-infinite" />;
  }
  return <span className={cn(isValidating && 'opacity-50')}>{datasource?.name ?? '(Unknown Datasource)'}</span>;
}
