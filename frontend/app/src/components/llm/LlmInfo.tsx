'use client';

import { useLlm } from '@/components/llm/hooks';
import { ModelComponentInfo } from '@/components/model-component-info';

export function LlmInfo ({ className, id }: { className?: string, id: number | undefined | null }) {
  const { llm, isLoading } = useLlm(id);

  return <ModelComponentInfo
    className={className}
    model={llm}
    url={llm => `/llms/${llm.id}`}
    isLoading={isLoading}
    defaultName="Default LLM"
  />;
}
