import type { ChatEngine } from '@/api/chat-engines';
import { type EmbeddingModel } from '@/api/embedding-models';
import type { EvaluationDataset } from '@/api/evaluations';
import { type KnowledgeBaseSummary } from '@/api/knowledge-base';
import { type LLM } from '@/api/llms';
import type { ProviderOption } from '@/api/providers';
import { type Reranker } from '@/api/rerankers';
import { useAllChatEngines } from '@/components/chat-engine/hooks';
import { DateFormat } from '@/components/date-format';
import { CreateEmbeddingModelForm } from '@/components/embedding-models/CreateEmbeddingModelForm';
import { useAllEmbeddingModels } from '@/components/embedding-models/hooks';
import { useAllEvaluationDatasets } from '@/components/evaluations/hooks';
import { FormCombobox, type FormComboboxConfig, type FormComboboxProps } from '@/components/form/control-widget';
import { useAllKnowledgeBases } from '@/components/knowledge-base/hooks';
import { CreateLLMForm } from '@/components/llm/CreateLLMForm';
import { useAllLlms } from '@/components/llm/hooks';
import { ManagedDialog } from '@/components/managed-dialog';
import { ManagedPanelContext } from '@/components/managed-panel';
import { CreateRerankerForm } from '@/components/reranker/CreateRerankerForm';
import { useAllRerankers } from '@/components/reranker/hooks';
import { Badge } from '@/components/ui/badge';
import { DialogContent, DialogDescription, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { AlertTriangleIcon, DotIcon, PlusIcon } from 'lucide-react';
import { forwardRef, type Ref } from 'react';

export const EmbeddingModelSelect = forwardRef<any, Omit<FormComboboxProps, 'config'> & { reverse?: boolean }>(({ reverse = true, ...props }, ref) => {
  const { data: embeddingModels, isLoading, mutate, error } = useAllEmbeddingModels();

  return (
    <FormCombobox
      {...props}
      ref={ref}
      placeholder="Default Embedding Model"
      config={{
        options: embeddingModels ?? [],
        optionKeywords: option => [option.name, option.provider, option.model],
        loading: isLoading,
        error,
        renderValue: option => (<span>{option.name} <span className="text-muted-foreground">[{option.vector_dimension}]</span></span>),
        renderOption: option => (
          <div>
            <div><strong>{option.name}</strong></div>
            <div className="text-xs text-muted-foreground">
              <strong>{option.provider}</strong>:{option.model} [{option.vector_dimension}]
            </div>
          </div>
        ),
        renderCreateOption: (Wrapper, onCreated) => (
          <ManagedDialog>
            <ManagedPanelContext.Consumer>
              {({ setOpen }) => (
                <>
                  <Wrapper onSelect={() => setOpen(true)}>
                    <span className="flex gap-1 items-center text-muted-foreground">
                      <PlusIcon className="size-4" />
                      Create New Embedding Model
                    </span>
                  </Wrapper>
                  <DialogContent>
                    <DialogHeader>
                      <DialogTitle>
                        Create New Embedding Model
                      </DialogTitle>
                      <DialogDescription />
                    </DialogHeader>
                    <CreateEmbeddingModelForm
                      onCreated={embeddingModel => {
                        mutate();
                        onCreated(embeddingModel);
                        setOpen(false);
                      }}
                    />
                  </DialogContent>
                </>
              )}
            </ManagedPanelContext.Consumer>
          </ManagedDialog>
        ),
        key: 'id',
      } satisfies FormComboboxConfig<EmbeddingModel>}
    />
  );
});

EmbeddingModelSelect.displayName = 'EmbeddingModelSelect';

export const LLMSelect = forwardRef<any, Omit<FormComboboxProps, 'config'> & { reverse?: boolean }>(({ reverse = true, ...props }, ref) => {
  const { data: llms, isLoading, mutate, error } = useAllLlms();

  return (
    <FormCombobox
      {...props}
      ref={ref}
      placeholder="Default LLM"
      config={{
        options: llms ?? [],
        loading: isLoading,
        error,
        renderValue: option => (<span>{option.name}</span>),
        renderOption: option => (
          <div>
            <div><strong>{option.name}</strong></div>
            <div className="text-xs text-muted-foreground">
              <strong>{option.provider}</strong>:{option.model}
            </div>
          </div>
        ),
        renderCreateOption: (Wrapper, onCreated) => (
          <ManagedDialog>
            <ManagedPanelContext.Consumer>
              {({ setOpen }) => (
                <>
                  <Wrapper onSelect={() => setOpen(true)}>
                    <span className="flex gap-1 items-center text-muted-foreground">
                      <PlusIcon className="size-4" />
                      Create New LLM
                    </span>
                  </Wrapper>
                  <DialogContent>
                    <DialogHeader>
                      <DialogTitle>
                        Create New LLM
                      </DialogTitle>
                      <DialogDescription />
                    </DialogHeader>
                    <CreateLLMForm
                      onCreated={llm => {
                        mutate();
                        onCreated(llm);
                        setOpen(false);
                      }}
                    />
                  </DialogContent>
                </>
              )}
            </ManagedPanelContext.Consumer>
          </ManagedDialog>
        ),
        optionKeywords: option => [option.name, option.provider, option.model],
        key: 'id',
      } satisfies FormComboboxConfig<LLM>}
    />
  );
});

LLMSelect.displayName = 'LLMSelect';

export const RerankerSelect = forwardRef<any, Omit<FormComboboxProps, 'config'> & { reverse?: boolean }>(({ reverse = true, ...props }, ref) => {
  const { data: rerankers, mutate, isLoading, error } = useAllRerankers();

  return (
    <FormCombobox
      {...props}
      ref={ref}
      placeholder="Default Reranker Model"
      config={{
        options: rerankers ?? [],
        optionKeywords: option => [option.name, option.provider, option.model],
        loading: isLoading,
        error,
        renderValue: option => (<span>{option.name}</span>),
        renderOption: option => (
          <div>
            <div><strong>{option.name}</strong></div>
            <div className="text-xs text-muted-foreground">
              <strong>{option.provider}</strong>:{option.model}
            </div>
          </div>
        ),
        renderCreateOption: (Wrapper, onCreated) => (
          <ManagedDialog>
            <ManagedPanelContext.Consumer>
              {({ setOpen }) => (
                <>
                  <Wrapper onSelect={() => setOpen(true)}>
                    <span className="flex gap-1 items-center text-muted-foreground">
                      <PlusIcon className="size-4" />
                      Create New Reranker
                    </span>
                  </Wrapper>
                  <DialogContent>
                    <DialogHeader>
                      <DialogTitle>
                        Create New Reranker
                      </DialogTitle>
                      <DialogDescription />
                    </DialogHeader>
                    <CreateRerankerForm
                      onCreated={reranker => {
                        mutate();
                        onCreated(reranker);
                        setOpen(false);
                      }}
                    />
                  </DialogContent>
                </>
              )}
            </ManagedPanelContext.Consumer>
          </ManagedDialog>
        ),
        key: 'id',
      } satisfies FormComboboxConfig<Reranker>}
    />
  );
});

RerankerSelect.displayName = 'RerankerSelect';

export interface ProviderSelectProps<Provider extends ProviderOption = ProviderOption> extends Omit<FormComboboxProps, 'config'> {
  options: ProviderOption[] | undefined;
  isLoading: boolean;
  error: unknown;
}

export const ProviderSelect = forwardRef<any, ProviderSelectProps>(({
  options, isLoading, error, ...props
}, ref) => {
  return (
    <FormCombobox
      ref={ref}
      config={{
        options: options ?? [],
        optionKeywords: option => [option.provider, option.provider_description ?? '', option.provider_display_name ?? ''],
        loading: isLoading,
        error,
        renderOption: option => (
          <div>
            <div className="text-sm font-bold">{option.provider_display_name ?? option.provider}</div>
            {option.provider_description && <div className="text-xs text-muted-foreground break-words" style={{ maxWidth: 'calc(var(--radix-select-trigger-width) - 68px)' }}>{option.provider_description}</div>}
          </div>
        ),
        itemClassName: 'space-y-1',
        renderValue: option => option.provider_display_name ?? option.provider,
        key: 'provider',
      } satisfies FormComboboxConfig<ProviderOption>}
      contentWidth="anchor"
      {...props}
    />
  );
});

ProviderSelect.displayName = 'ProviderSelect';

export const KBSelect = forwardRef<any, Omit<FormComboboxProps, 'config'> & { reverse?: boolean }>(({ reverse = true, ...props }, ref) => {
  const { data: kbs, isLoading, error } = useAllKnowledgeBases();

  return (
    <FormCombobox
      ref={ref}
      {...props}
      placeholder="Select Knowledge Base"
      config={{
        options: kbs ?? [],
        optionKeywords: option => [String(option.id), option.name, option.description],
        loading: isLoading,
        error,
        renderValue: option => (
          <div className="">
            <span>{option.name}</span>
            <div className="text-xs text-muted-foreground ml-2 inline-flex gap-1 items-center">
              <span>
                {(option.documents_total ?? 0) || <><AlertTriangleIcon className="text-warning inline-flex size-3 mr-0.5" /> no</>} documents
              </span>
              <DotIcon className="size-4" />
              <span className="text-xs text-muted-foreground">
                {(option.data_sources_total ?? 0) || <><AlertTriangleIcon className="inline-flex size-3 mr-0.5" /> no</>} data sources
              </span>
            </div>
          </div>
        ),
        renderOption: option => (
          <div className="space-y-1">
            <div>
              <strong>
                {option.name}
              </strong>
            </div>
            <div className="text-xs text-muted-foreground flex gap-1 items-center">
              <span>
                {(option.documents_total ?? 0) || <><AlertTriangleIcon className="text-warning inline-flex size-3 mr-0.5" /> no</>} documents
              </span>
              <DotIcon className="size-4" />
              <span>
                {(option.data_sources_total ?? 0) || <><AlertTriangleIcon className="inline-flex size-3 mr-0.5" /> no</>} data sources
              </span>
            </div>
            <div className="text-xs text-muted-foreground">
              {option.description}
            </div>
          </div>
        ),
        key: 'id',
      } satisfies FormComboboxConfig<KnowledgeBaseSummary>}
    />
  );
});

KBSelect.displayName = 'KBSelect';

export function EvaluationDatasetSelect ({ reverse = true, ref, ...props }: Omit<FormComboboxProps, 'config'> & { reverse?: boolean, ref?: Ref<any> }) {

  const { data: evaluationDatasets, isLoading, error } = useAllEvaluationDatasets();

  return (
    <FormCombobox
      {...props}
      ref={ref}
      placeholder="Select Evaluation Dataset"
      config={{
        options: evaluationDatasets ?? [],
        optionKeywords: option => [option.name],
        loading: isLoading,
        error,
        renderValue: option => (<span>{option.name}</span>),
        renderOption: option => (
          <div>
            <div><strong>{option.name}</strong></div>
            <div className="text-xs text-muted-foreground">
              Updated At: <DateFormat date={option.updated_at} />
            </div>
          </div>
        ),
        key: 'id',
      } satisfies FormComboboxConfig<EvaluationDataset>}
    />
  );
}

export function ChatEngineSelect ({ reverse = true, ref, ...props }: Omit<FormComboboxProps, 'config'> & { reverse?: boolean, ref?: Ref<any> }) {
  const { data: chatEngines, isLoading, error } = useAllChatEngines();

  return (
    <FormCombobox
      {...props}
      ref={ref}
      placeholder="Default Chat Engine"
      config={{
        options: chatEngines ?? [],
        optionKeywords: option => [option.name],
        loading: isLoading,
        error,
        renderValue: option => (
          <>
            <strong>{option.name}</strong>
            {!!option.engine_options.external_engine_config?.stream_chat_api_url && <Badge className="ml-2 font-normal" variant="secondary">External Chat Engine</Badge>}
            {!!option.engine_options.knowledge_graph?.enabled && <Badge className="ml-2 font-normal" variant="secondary">Knowledge Graph</Badge>}
          </>
        ),
        renderOption: option => (
          <div>
            <strong>{option.name}</strong>
            {option.is_default && <Badge className="ml-2">Default</Badge>}
            {!!option.engine_options.external_engine_config?.stream_chat_api_url && <Badge className="ml-2 font-normal" variant="secondary">External Chat Engine</Badge>}
            {!!option.engine_options.knowledge_graph?.enabled && <Badge className="ml-2 font-normal" variant="secondary">Knowledge Graph</Badge>}
          </div>
        ),
        key: 'name',
      } satisfies FormComboboxConfig<ChatEngine>}
    />
  );
}
