import { EntitiesTable, entityColumns } from '@/components/graph/components/EntitiesTable';
import { type UseEntitiesRequired, useGraphEntitiesByIdsTable } from '@/components/graph/selectEntities';
import { Button } from '@/components/ui/button';
import { Dialog, DialogContent, DialogHeader, DialogPortal, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { Form, FormControl, FormField } from '@/components/ui/form';
import { Input } from '@/components/ui/input';
import { cn } from '@/lib/utils';
import { zodResolver } from '@hookform/resolvers/zod';
import isHotkey from 'is-hotkey';
import { PlusIcon } from 'lucide-react';
import { useForm } from 'react-hook-form';
import z from 'zod';

export interface SearchEntityProps extends UseEntitiesRequired {
  knowledgeBaseId: number
}

/**
 * @deprecated
 */
export function SearchEntityById ({ knowledgeBaseId, ...props }: SearchEntityProps) {
  const { table, ids, setIds } = useGraphEntitiesByIdsTable(knowledgeBaseId, {
    columns: entityColumns,
    ...props,
  });

  return (
    <Dialog>
      <DialogTrigger asChild>
        <Button variant='secondary'>
          <PlusIcon className="mr-2 w-3.5 h-3.5" />
          Entities by IDs
        </Button>
      </DialogTrigger>
      <DialogPortal>
        <DialogContent className="overflow-x-hidden max-w-screen-md space-y-2">
          <DialogHeader>
            <DialogTitle>
              Search entities by IDs
            </DialogTitle>
          </DialogHeader>
          <TableFilterForm filter={ids} onFilterChange={setIds} />
          <EntitiesTable table={table} isLoading={false} />
        </DialogContent>
      </DialogPortal>
    </Dialog>
  );
}

function TableFilterForm ({ className, filter, onFilterChange, disabled }: { className?: string, filter: number[], onFilterChange: (filter: number[]) => void, disabled?: boolean }) {
  const form = useForm<{ ids: string }>({
    values: { ids: filter.map(i => String(i)).join(', ') },
    resolver: zodResolver(tableFilterSchema),
    disabled,
    defaultValues: {
      ids: '',
    },
  });

  const populate = () => {
    const ids = form.getValues().ids.split(',')
      .map(s => s.trim())
      .filter(Boolean)
      .map(s => parseInt(s))
      .filter(n => isFinite(n));
    onFilterChange(ids);
  };

  return (
    <Form {...form}>
      <form className={cn('flex gap-2 items-center', className)} onSubmit={event => {
        event.stopPropagation();
        event.preventDefault();
        return false;
      }}>
        <FormField
          name="ids"
          render={({ field }) => (
            <FormControl>
              <Input
                {...field}
                placeholder="Input entities ids separated by ','"
                onKeyDown={ev => {
                  if (isHotkey('Enter', ev)) {
                    ev.stopPropagation();
                    ev.preventDefault();
                    populate();
                  }
                }}
              />
            </FormControl>
          )}
        />
      </form>
    </Form>
  );
}

const tableFilterSchema = z.object({
  ids: z.string(),
});