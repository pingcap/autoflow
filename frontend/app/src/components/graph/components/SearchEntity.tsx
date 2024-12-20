import { EntitiesTable, entityColumns } from '@/components/graph/components/EntitiesTable';
import { type SearchEntityFilter, type UseEntitiesRequired, useGraphEntitiesTable } from '@/components/graph/selectEntities';
import { Button } from '@/components/ui/button';
import { Dialog, DialogContent, DialogHeader, DialogPortal, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { Form, FormControl, FormField } from '@/components/ui/form';
import { Input } from '@/components/ui/input';
import { cn } from '@/lib/utils';
import { zodResolver } from '@hookform/resolvers/zod';
import isHotkey from 'is-hotkey';
import { SearchIcon } from 'lucide-react';
import { useForm } from 'react-hook-form';
import z from 'zod';

export interface SearchEntityProps extends UseEntitiesRequired {
  knowledgeBaseId: number
}

/**
 * @deprecated
 */
export function SearchEntity ({ knowledgeBaseId, ...props }: SearchEntityProps) {
  const { table, filter, setFilter, isLoading, error } = useGraphEntitiesTable(knowledgeBaseId, {
    columns: entityColumns,
    ...props,
  });

  return (
    <Dialog>
      <DialogTrigger asChild>
        <Button variant='secondary'>
          <SearchIcon className="mr-2 w-3.5 h-3.5" />
          Search entities
        </Button>
      </DialogTrigger>
      <DialogPortal>
        <DialogContent className="overflow-x-hidden max-w-screen-md space-y-2">
          <DialogHeader>
            <DialogTitle>
              Search entities
            </DialogTitle>
          </DialogHeader>
          <TableFilterForm filter={filter} onFilterChange={setFilter} />
          <EntitiesTable table={table} isLoading={isLoading} />
        </DialogContent>
      </DialogPortal>
    </Dialog>
  );
}

function TableFilterForm ({ className, filter, onFilterChange, disabled }: { className?: string, filter: SearchEntityFilter, onFilterChange: (filter: SearchEntityFilter) => void, disabled?: boolean }) {
  const form = useForm<SearchEntityFilter>({
    values: filter,
    resolver: zodResolver(tableFilterSchema),
    disabled,
    defaultValues: {
      query: '',
      top_k: undefined,
    },
  });

  return (
    <Form {...form}>
      <form className={cn('flex gap-2 items-center', className)} onSubmit={event => {
        event.stopPropagation();
        event.preventDefault();
        return false;
      }}>
        <FormField
          name="query"
          render={({ field }) => (
            <FormControl>
              <Input
                {...field}
                placeholder="Input your query..."
                onKeyDown={ev => {
                  if (isHotkey('Enter', ev)) {
                    ev.stopPropagation();
                    ev.preventDefault();
                    onFilterChange(form.getValues());
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
  query: z.string(),
  top_k: z.coerce.number().optional(),
});