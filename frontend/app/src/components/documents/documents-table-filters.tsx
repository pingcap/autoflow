import { type Document, listDocumentsFiltersSchema, type ListDocumentsTableFilters, mimeTypes } from '@/api/documents';
import { indexStatuses } from '@/api/rag';
import { Button } from '@/components/ui/button';
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible';
import { Form, FormControl, formDomEventHandlers, FormField, FormItem, FormLabel, FormMessage } from '@/components/ui/form.beta';
import { Input } from '@/components/ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { useForm } from '@tanstack/react-form';
import { Table as ReactTable } from '@tanstack/react-table';
import { capitalCase } from 'change-case-all';
import { ChevronDownIcon, UploadIcon } from 'lucide-react';
import { DateRangePicker } from '@/components/date-range-picker';
import { DateRange } from 'react-day-picker';
import { NextLink } from '@/components/nextjs/NextLink';

interface DocumentsTableFiltersProps {
  knowledgeBaseId: number;
  table: ReactTable<Document>;
  onFilterChange: (data: ListDocumentsTableFilters) => void;
}

export function DocumentsTableFilters ({ knowledgeBaseId, table, onFilterChange }: DocumentsTableFiltersProps) {
  const form = useForm({
    validators: {
      onChange: listDocumentsFiltersSchema,
    },
    defaultValues: {
      search: undefined,
      mime_type: undefined,
      index_status: undefined,
    },
    onSubmit: async ({ value }) => {
      const filters = listDocumentsFiltersSchema.parse(value);
      onFilterChange?.(filters);
    },
  });

  return (
    <Form form={form}>
      <div className="flex flex-col gap-4">
        {/* Top row - Search and Upload */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <FormField
              name="search"
              render={(field) => (
                <FormItem>
                  <FormControl>
                    <Input
                      name={field.name}
                      className="h-8 text-sm w-[300px]"
                      onBlur={field.handleBlur}
                      onChange={ev => field.handleChange(ev.target.value)}
                      value={field.state.value ?? ''}
                      placeholder="Search documents"
                      onKeyDown={(e) => {
                        if (e.key === 'Enter') {
                          e.preventDefault();
                          form.handleSubmit();
                        }
                      }}
                    />
                  </FormControl>
                </FormItem>
              )}
            />
            <Button 
              type="submit" 
              size="sm" 
              className="h-8 px-3"
              onClick={(e) => {
                e.preventDefault();
                form.handleSubmit();
              }}
            >
              Search
            </Button>
          </div>
          <div className="flex items-center gap-2">
            <NextLink
              href={`/knowledge-bases/${knowledgeBaseId}/data-sources/new?type=file`}
              variant="secondary"
              className="h-8 text-sm px-3"
            >
              <UploadIcon className="mr-2 size-3" />
              Upload
            </NextLink>
          </div>
        </div>

        {/* Bottom row - Filters */}
        <div className="flex items-center gap-2 flex-wrap">

          <FormField
            name="mime_type"
            render={(field) => (
              <FormItem>
                <Select value={field.state.value ?? ''} name={field.name} onValueChange={field.handleChange}>
                  <SelectTrigger className="h-8 text-sm font-normal hover:bg-accent" onBlur={field.handleBlur}>
                    <SelectValue placeholder="Document Type" />
                  </SelectTrigger>
                  <SelectContent>
                    {mimeTypes.map(mime => (
                      <SelectItem key={mime.value} value={mime.value}>
                        {mime.name}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </FormItem>
            )}
          />

          <FormField
            name="index_status"
            render={(field) => (
              <FormItem>
                <Select value={field.state.value ?? ''} name={field.name} onValueChange={field.handleChange}>
                  <SelectTrigger className="h-8 text-sm font-normal hover:bg-accent" onBlur={field.handleBlur}>
                    <SelectValue placeholder="Index Status" />
                  </SelectTrigger>
                  <SelectContent>
                    {indexStatuses.map(indexStatus => (
                      <SelectItem key={indexStatus} value={indexStatus}>
                        {capitalCase(indexStatus)}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </FormItem>
            )}
          />

          <FormField
            name="created_at"
            render={(field) => (
              <FormItem>
                <DateRangePicker
                  value={field.state.value ? { from: field.state.value[0], to: field.state.value[1] } : undefined}
                  onChange={(range) => field.handleChange(range ? [range.from, range.to] : undefined)}
                  placeholder="Created Time"
                  size="sm"
                />
              </FormItem>
            )}
          />

          <FormField
            name="updated_at"
            render={(field) => (
              <FormItem>
                <DateRangePicker
                  value={field.state.value ? { from: field.state.value[0], to: field.state.value[1] } : undefined}
                  onChange={(range) => field.handleChange(range ? [range.from, range.to] : undefined)}
                  placeholder="Updated Time"
                  size="sm"
                />
              </FormItem>
            )}
          />

          <FormField
            name="last_modified_at"
            render={(field) => (
              <FormItem>
                <DateRangePicker
                  value={field.state.value ? { from: field.state.value[0], to: field.state.value[1] } : undefined}
                  onChange={(range) => field.handleChange(range ? [range.from, range.to] : undefined)}
                  placeholder="Last Modified Time"
                  size="sm"
                />
              </FormItem>
            )}
          />

          <Button 
            variant="ghost" 
            className="text-sm font-normal h-8 px-2 hover:bg-accent"
            onClick={() => form.reset()}
          >
            Clear filters
          </Button>
        </div>
      </div>
    </Form>
  );
}