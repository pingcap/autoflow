import { type EmbeddingModel, listEmbeddingModels } from '@/api/embedding-models';
import { listLlms, type LLM } from '@/api/llms';
import type { ProviderOption } from '@/api/providers';
import { listRerankers, type Reranker } from '@/api/rerankers';
import { EmbeddingModelInfo } from '@/components/embedding-models/EmbeddingModelInfo';
import { FormSelect, type FormSelectConfig, type FormSelectProps } from '@/components/form/control-widget';
import { LlmInfo } from '@/components/llm/LlmInfo';
import { RerankerInfo } from '@/components/reranker/RerankerInfo';
import { forwardRef } from 'react';
import useSWR from 'swr';

export const EmbeddingModelSelect = forwardRef<any, Omit<FormSelectProps, 'config'> & { reverse?: boolean }>(({ reverse = true, ...props }, ref) => {
  // TODO
  const { data: embeddingModels, isLoading, error } = useSWR('api.embedding-models.list-all', () => listEmbeddingModels({ size: 100 }));

  return (
    <FormSelect
      {...props}
      placeholder="Default Embedding Model"
      config={{
        options: embeddingModels?.items ?? [],
        loading: isLoading,
        error,
        renderValue: option => (<span><EmbeddingModelInfo reverse={reverse} id={option.id} /></span>),
        renderOption: option => (<span><EmbeddingModelInfo detailed reverse={reverse} id={option.id} /></span>),
        key: 'id',
      } satisfies FormSelectConfig<EmbeddingModel>}
    />
  );
});

EmbeddingModelSelect.displayName = 'EmbeddingModelSelect';

export const LLMSelect = forwardRef<any, Omit<FormSelectProps, 'config'> & { reverse?: boolean }>(({ reverse = true, ...props }, ref) => {
  const { data: llms, isLoading, error } = useSWR('api.llms.list-all', () => listLlms({ size: 100 }));

  return (
    <FormSelect
      {...props}
      placeholder="Default LLM"
      config={{
        options: llms?.items ?? [],
        loading: isLoading,
        error,
        renderValue: option => (<span><LlmInfo reverse={reverse} id={option.id} /></span>),
        renderOption: option => (<span><LlmInfo detailed reverse={reverse} id={option.id} /></span>),
        key: 'id',
      } satisfies FormSelectConfig<LLM>}
    />
  );
});

LLMSelect.displayName = 'LLMSelect';

export const RerankerSelect = forwardRef<any, Omit<FormSelectProps, 'config'> & { reverse?: boolean }>(({ reverse = true, ...props }, ref) => {
  const { data: rerankers, isLoading, error } = useSWR('api.rerankers.list-all', () => listRerankers({ size: 100 }));

  return (
    <FormSelect
      {...props}
      placeholder="Default Reranker Model"
      config={{
        options: rerankers?.items ?? [],
        loading: isLoading,
        error,
        renderValue: option => (<span><RerankerInfo reverse={reverse} id={option.id} /></span>),
        renderOption: option => (<span><RerankerInfo detailed reverse={reverse} id={option.id} /></span>),
        key: 'id',
      } satisfies FormSelectConfig<Reranker>}
    />
  );
});

RerankerSelect.displayName = 'RerankerSelect';

export interface ProviderSelectProps<Provider extends ProviderOption = ProviderOption> extends Omit<FormSelectProps, 'config'> {
  options: ProviderOption[] | undefined;
  isLoading: boolean;
  error: unknown;
}

export const ProviderSelect = forwardRef<any, ProviderSelectProps>(({
  options, isLoading, error, ...props
}, ref) => {
  return (
    <FormSelect
      ref={ref}
      config={{
        options: options ?? [],
        loading: isLoading,
        error,
        renderOption: option => (
          <>
            <div className="text-sm font-bold max-w-screen-sm">{option.provider_display_name ?? option.provider}</div>
            {option.provider_description && <div className="text-xs text-muted-foreground break-words" style={{ maxWidth: 'calc(var(--radix-select-trigger-width) - 68px)' }}>{option.provider_description}</div>}
          </>
        ),
        itemClassName: 'space-y-1',
        renderValue: option => option.provider_display_name ?? option.provider,
        key: 'provider',
      } satisfies FormSelectConfig<ProviderOption>}
      {...props}
    />
  );
});

ProviderSelect.displayName = 'ProviderSelect';
