'use client';

import { useEmbeddingModel } from '@/components/embedding-models/hooks';
import { ModelComponentInfo } from '@/components/model-component-info';

export function EmbeddingModelInfo ({ className, reverse = false, detailed = false, id }: { className?: string, reverse?: boolean, detailed?: boolean, id: number | undefined | null }) {
  const { embeddingModel, isLoading } = useEmbeddingModel(id);

  return <ModelComponentInfo
    className={className}
    model={embeddingModel}
    url={embeddingModel => `/embedding-models/${embeddingModel.id}`}
    isLoading={isLoading}
    reverse={reverse}
    detailed={detailed}
    defaultName="Default Embedding Model"
  />;
}
