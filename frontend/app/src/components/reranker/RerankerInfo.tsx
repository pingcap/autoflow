'use client';

import { ModelComponentInfo } from '@/components/model-component-info';
import { useReranker } from '@/components/reranker/hooks';

export function RerankerInfo ({ className, id }: { className?: string, id: number | undefined | null }) {
  const { reranker, isLoading } = useReranker(id);

  return <ModelComponentInfo
    className={className}
    model={reranker}
    url={reranker => `/reranker-models/${reranker.id}`}
    isLoading={isLoading}
    defaultName="Default Reranker Model"
  />;
}
