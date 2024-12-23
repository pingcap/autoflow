import { createSynopsisEntity, type KnowledgeGraphEntity } from '@/api/graph';
import { SearchEntity } from '@/components/graph/components/SearchEntity';
import { SearchEntityById } from '@/components/graph/components/SearchEntityById';
import { useEntities } from '@/components/graph/selectEntities';
import { Loader } from '@/components/loader';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Form, FormControl, FormField, FormItem, FormLabel, FormMessage } from '@/components/ui/form';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';
import { cn } from '@/lib/utils';
import { zodResolver } from '@hookform/resolvers/zod';
import { Loader2Icon, Maximize2Icon } from 'lucide-react';
import type monaco from 'monaco-editor';
import { lazy, type ReactNode, Suspense, useRef } from 'react';
import { useForm } from 'react-hook-form';
import z from 'zod';

const JsonEditor = lazy(() => import('./components/JsonEditor').then(res => ({ default: res.JsonEditor })));

/**
 * @deprecated
 */
export function GraphCreateEntity ({ className, knowledgeBaseId, onCreated }: { className?: string, knowledgeBaseId: number, onCreated: (entity: KnowledgeGraphEntity) => void }) {
  const useEntitiesReturns = useEntities();
  const { clearSelection, ...useEntitiesRequired } = useEntitiesReturns;
  const { selectedEntities } = useEntitiesRequired;

  return (
    <div className="space-y-4">
      <div className="lg:max-w-[50vw]">
        <CreateEntityForm
          entities={selectedEntities}
          onSubmit={async (data) => {
            const createdEntity = await createSynopsisEntity(knowledgeBaseId, data);
            onCreated(createdEntity);
          }}
          onClearSelection={clearSelection}
          afterEntities={(
            <>
              <SearchEntity knowledgeBaseId={knowledgeBaseId} {...useEntitiesRequired} />
              <SearchEntityById knowledgeBaseId={knowledgeBaseId} {...useEntitiesRequired} />
            </>
          )}
        />
      </div>
    </div>
  );
}

function CreateEntityForm ({ className, entities, onSubmit, onClearSelection, afterEntities }: { className?: string, entities: KnowledgeGraphEntity[], onSubmit: (data: z.infer<typeof createEntitySchema> & { meta: any, entities: number[] }) => Promise<void>, onClearSelection: (id?: number) => void, afterEntities?: ReactNode }) {
  const form = useForm<z.infer<typeof createEntitySchema>>({
    resolver: zodResolver(createEntitySchema),
    defaultValues: {
      name: '',
      description: '',
      topic: '',
    },
  });
  const metaRef = useRef<monaco.editor.IStandaloneCodeEditor | undefined | null>(null);

  const handleSubmit = form.handleSubmit(async values => onSubmit({
    ...values,
    meta: JSON.parse(metaRef.current!.getValue()),
    entities: entities.map(entity => Number(entity.id)),
  }));

  const containerRef = useRef<HTMLDivElement>(null);

  const handleClickFullscreen = () => {
    containerRef.current?.requestFullscreen();
  };

  return (
    <Form {...form}>
      <form className={cn('space-y-4', className)} onSubmit={handleSubmit}>
        <h2 className="font-bold text-xl">Create synopsis entity</h2>
        <FormField
          name="name"
          render={({ field }) => (
            <FormItem>
              <FormLabel>Name</FormLabel>
              <FormControl>
                <Input {...field} />
              </FormControl>
              <FormMessage />
            </FormItem>
          )}
        />
        <FormField
          name="description"
          render={({ field }) => (
            <FormItem>
              <FormLabel>Description</FormLabel>
              <FormControl>
                <Textarea {...field} />
              </FormControl>
              <FormMessage />
            </FormItem>
          )}
        />
        <FormField
          name="topic"
          render={({ field }) => (
            <FormItem>
              <FormLabel>Topic</FormLabel>
              <FormControl>
                <Input {...field} />
              </FormControl>
              <FormMessage />
            </FormItem>
          )}
        />
        <FormItem>
          <div className="flex items-center justify-between">
            <Label>
              Meta
            </Label>
            <button className="text-foreground/50 hover:text-foreground transition-colors" onMouseDown={handleClickFullscreen}>
              <Maximize2Icon className="w-3 h-3" />
            </button>
          </div>
          <FormControl>
            <div className="relative w-full h-32 border" ref={containerRef}>
              <Suspense fallback={<Loader loading>Initializing JSON editor...</Loader>}>
                <JsonEditor defaultValue="{}" ref={metaRef} />
              </Suspense>
            </div>
          </FormControl>
        </FormItem>
        <FormItem>
          <Label>Entities</Label>
          <TooltipProvider>
            <div className="flex gap-2 flex-wrap">
              {entities.map(entity => (
                <Tooltip key={entity.id}>
                  <TooltipTrigger type="button">
                    <Badge key={entity.id} variant="secondary">{entity.name} #{entity.id}</Badge>
                  </TooltipTrigger>
                  <TooltipContent className="space-y-2 w-[360px]">
                    <h3 className="font-bold">{entity.name} #{entity.id}</h3>
                    <p className="text-xs text-accent-foreground">{entity.description}</p>
                    <Button variant="secondary" className="w-full mt-4" size="sm" onClick={() => onClearSelection(Number(entity.id))}>Remove from entities</Button>
                  </TooltipContent>
                </Tooltip>
              ))}
            </div>
          </TooltipProvider>
          <div className="grid grid-cols-3 gap-2">
            {afterEntities}
            <Button variant="ghost" onClick={() => onClearSelection()}>
              Clear Selection
            </Button>
          </div>
        </FormItem>
        <div className="!mt-8">
          <Button type="submit" disabled={form.formState.disabled}>
            {form.formState.isSubmitting && <Loader2Icon className="w-4 h-4 mr-2 animate-spin repeat-infinite" />}
            Create Entity
          </Button>
        </div>
      </form>
    </Form>
  );
}

const createEntitySchema = z.object({
  name: z.string().min(1).regex(/\S/),
  description: z.string().min(1).regex(/\S/),
  topic: z.string().min(1).regex(/\S/),
});
