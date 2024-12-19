import { Button } from '@/components/ui/button';
import { Form, FormControl, FormField } from '@/components/ui/form';
import { Input } from '@/components/ui/input';
import { useDataTable } from '@/components/use-data-table';
import { zodResolver } from '@hookform/resolvers/zod';
import { useForm } from 'react-hook-form';
import { z } from 'zod';

export function KeywordFilterToolbar ({ onFilterChange }: { onFilterChange: (filters: KeywordFilter) => void }) {
  const { loading } = useDataTable();

  const form = useForm({
    resolver: zodResolver(keywordFilter),
    defaultValues: {
      keyword: '',
    },
    disabled: loading,
  });

  const handleSubmit = form.handleSubmit(({ keyword, ...rest }) => {
    const trimmedKeyword = keyword.trim();
    onFilterChange({
      keyword: trimmedKeyword ? trimmedKeyword : undefined,
      ...rest,
    });
  });

  return (
    <Form {...form}>
      <form className="flex gap-2 items-center" onSubmit={handleSubmit}>
        <FormField
          name="keyword"
          render={({ field }) => (
            <FormControl>
              <Input
                className="flex-1"
                placeholder="Search Evaluation Datasets..."
                {...field}
              />
            </FormControl>
          )}
        />
        <Button variant="secondary" disabled={form.formState.disabled} type="submit">
          Search
        </Button>
      </form>
    </Form>
  );
}

const keywordFilter = z.object({
  keyword: z.string().optional(),
});

export type KeywordFilter = z.infer<typeof keywordFilter>;
