'use client';

import { useEmbeddingModel } from '@/components/embedding-models/hooks';
import { ModelComponentInfo } from '@/components/model-component-info';

export function EmbeddingModelInfo ({ className, id }: { className?: string, id: number | undefined | null }) {
  const { embeddingModel, isLoading } = useEmbeddingModel(id);

  return <ModelComponentInfo
    className={className}
    model={embeddingModel}
    url={embeddingModel => `/embedding-models/${embeddingModel.id}`}
    isLoading={isLoading}
    defaultName="Default Embedding Model"
  />;
}
