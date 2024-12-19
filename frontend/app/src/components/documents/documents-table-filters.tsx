import { type Document, listDocumentsFiltersSchema, type ListDocumentsTableFilters, mimeTypes } from '@/api/documents';
import { indexStatuses } from '@/api/rag';
import { Button } from '@/components/ui/button';
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible';
import { Form, FormControl, FormField, FormItem, FormLabel, FormMessage } from '@/components/ui/form';
import { Input } from '@/components/ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { zodResolver } from '@hookform/resolvers/zod';
import { Table as ReactTable } from '@tanstack/react-table';
import { capitalCase } from 'change-case-all';
import { ChevronDownIcon } from 'lucide-react';
import { useForm } from 'react-hook-form';

export function DocumentsTableFilters ({ onFilterChange }: { table: ReactTable<Document>, onFilterChange: (data: ListDocumentsTableFilters) => void }) {
  const form = useForm({
    resolver: zodResolver(listDocumentsFiltersSchema),
  });

  const onSubmit = form.handleSubmit((data) => {
    onFilterChange?.(data);
  });

  return (
    <Form {...form}>
      <Collapsible asChild>
        <form className="space-y-4" onSubmit={onSubmit}>
          <div className="flex gap-2 items-center">
            <FormField
              name="name"
              render={({ field }) => (
                <FormItem className='flex-1'>
                  <FormControl>
                    <Input {...field} value={field.value ?? ''} placeholder="Search..." />
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />
            <CollapsibleTrigger className="group text-sm flex items-center py-1.5 hover:underline focus:underline outline-none">
              Advanced Filters
              <ChevronDownIcon className="size-4 mr-1 transition-transform group-data-[state=open]:rotate-180" />
            </CollapsibleTrigger>
          </div>
          <CollapsibleContent className="py-2 space-y-4">
            <FormField
              name="source_uri"
              render={({ field }) => (
                <FormItem>
                  <FormControl>
                    <Input {...field} value={field.value ?? ''} placeholder="Search Source URI..." />
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />
            <div className="grid grid-cols-2 gap-4">
              <FormField
                name="mime_type"
                render={({ field: { value, onChange, name, disabled, ...field } }) => (
                  <FormItem>
                    <FormControl>
                      <Select value={value ?? ''} name={name} disabled={disabled} onValueChange={onChange}>
                        <SelectTrigger {...field}>
                          <SelectValue placeholder={<span className="text-muted-foreground">Select Document Type...</span>} />
                        </SelectTrigger>
                        <SelectContent>
                          {mimeTypes.map(mime => (
                            <SelectItem key={mime.value} value={mime.value}>
                              {mime.name}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </FormControl>
                    <FormMessage />
                  </FormItem>
                )}
              />
              <FormField
                name="index_status"
                render={({ field: { value, onChange, name, disabled, ...field } }) => (
                  <FormItem>
                    <FormControl>
                      <Select value={value ?? ''} name={name} disabled={disabled} onValueChange={onChange}>
                        <SelectTrigger {...field}>
                          <SelectValue placeholder={<span className="text-muted-foreground">Select Index Status...</span>} />
                        </SelectTrigger>
                        <SelectContent>
                          {indexStatuses.map(indexStatus => (
                            <SelectItem key={indexStatus} value={indexStatus}>
                              {capitalCase(indexStatus)}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </FormControl>
                    <FormMessage />
                  </FormItem>
                )}
              />
              <FormField
                name="created_at_start"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Created After</FormLabel>
                    <FormControl>
                      <Input {...field} value={field.value ?? ''} type="datetime-local" />
                    </FormControl>
                    <FormMessage />
                  </FormItem>
                )}
              />
              <FormField
                name="created_at_end"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Created Before</FormLabel>
                    <FormControl>
                      <Input {...field} value={field.value ?? ''} type="datetime-local" />
                    </FormControl>
                    <FormMessage />
                  </FormItem>
                )}
              />
              <FormField
                name="updated_at_start"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Updated After</FormLabel>
                    <FormControl>
                      <Input {...field} value={field.value ?? ''} type="datetime-local" />
                    </FormControl>
                    <FormMessage />
                  </FormItem>
                )}
              />
              <FormField
                name="updated_at_end"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Updated Before</FormLabel>
                    <FormControl>
                      <Input {...field} value={field.value ?? ''} type="datetime-local" />
                    </FormControl>
                    <FormMessage />
                  </FormItem>
                )}
              />
              <FormField
                name="last_modified_at_start"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Last Modified After</FormLabel>
                    <FormControl>
                      <Input {...field} value={field.value ?? ''} type="datetime-local" />
                    </FormControl>
                    <FormMessage />
                  </FormItem>
                )}
              />
              <FormField
                name="last_modified_at_end"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Last Modified Before</FormLabel>
                    <FormControl>
                      <Input {...field} value={field.value ?? ''} type="datetime-local" />
                    </FormControl>
                    <FormMessage />
                  </FormItem>
                )}
              />
            </div>
          </CollapsibleContent>
          <Button type="submit">Search</Button>
        </form>
      </Collapsible>
    </Form>
  );
}