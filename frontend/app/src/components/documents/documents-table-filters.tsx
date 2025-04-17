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
import { ChevronDownIcon } from 'lucide-react';

export function DocumentsTableFilters ({ onFilterChange }: { table: ReactTable<Document>, onFilterChange: (data: ListDocumentsTableFilters) => void }) {
  const form = useForm({
    validators: {
      onChange: listDocumentsFiltersSchema,
    },
    onSubmit: ({ value }) => {
      onFilterChange?.(listDocumentsFiltersSchema.parse(value));
    },
  });

  return (
    <Form form={form}>
      <div className="flex items-center">
        <div className="flex items-center gap-2">
          <FormField
            name="search"
            render={(field) => (
              <FormItem>
                <FormControl>
                  <div className="flex gap-2">
                    <Input
                      name={field.name}
                      className="h-8 text-sm w-[400px]"
                      onBlur={field.handleBlur}
                      onChange={ev => field.handleChange(ev.target.value)}
                      value={field.state.value ?? ''}
                      placeholder="Search..."
                    />
                    <Button 
                      type="submit" 
                      size="sm" 
                      className="h-8 px-3"
                      onClick={() => form.handleSubmit()}
                    >
                      Search
                    </Button>
                  </div>
                </FormControl>
              </FormItem>
            )}
          />

          <FormField
            name="mime_type"
            render={(field) => (
              <FormItem>
                <Select value={field.state.value ?? ''} name={field.name} onValueChange={field.handleChange}>
                  <SelectTrigger className="h-8 text-sm font-normal hover:bg-accent min-w-[120px]" onBlur={field.handleBlur}>
                    <SelectValue placeholder="Type" />
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
                  <SelectTrigger className="h-8 text-sm font-normal hover:bg-accent min-w-[120px]" onBlur={field.handleBlur}>
                    <SelectValue placeholder="Status" />
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
            name="source_uri"
            render={(field) => (
              <FormItem>
                <FormControl>
                  <Input
                    name={field.name}
                    className="h-8 text-sm w-[180px]"
                    onBlur={field.handleBlur}
                    onChange={ev => field.handleChange(ev.target.value)}
                    value={field.state.value ?? ''}
                    placeholder="Source URI"
                  />
                </FormControl>
              </FormItem>
            )}
          />

          <Collapsible>
            <CollapsibleTrigger asChild>
              <Button variant="ghost" size="sm" className="h-8 px-2 font-normal">
                More filters
                <ChevronDownIcon className="ml-1 size-4 transition-transform group-data-[state=open]:rotate-180" />
              </Button>
            </CollapsibleTrigger>
            <CollapsibleContent className="absolute mt-2 p-4 bg-background border rounded-md shadow-md z-10">
              <div className="grid grid-cols-2 gap-4 min-w-[600px]">
                <FormField
                  name="created_at_start"
                  render={(field) => (
                    <FormItem>
                      <FormLabel>Created After</FormLabel>
                      <FormControl>
                        <Input
                          name={field.name}
                          onBlur={field.handleBlur}
                          onChange={ev => field.handleChange(ev.target.valueAsDate)}
                          type="datetime-local"
                          value={field.state.value ?? ''}
                        />
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />
                <FormField
                  name="created_at_end"
                  render={(field) => (
                    <FormItem>
                      <FormLabel>Created Before</FormLabel>
                      <FormControl>
                        <Input
                          name={field.name}
                          onBlur={field.handleBlur}
                          onChange={ev => field.handleChange(ev.target.valueAsDate)}
                          type="datetime-local"
                          value={field.state.value ?? ''}
                        />
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />
                <FormField
                  name="updated_at_start"
                  render={(field) => (
                    <FormItem>
                      <FormLabel>Updated After</FormLabel>
                      <FormControl>
                        <Input
                          name={field.name}
                          onBlur={field.handleBlur}
                          onChange={ev => field.handleChange(ev.target.valueAsDate)}
                          type="datetime-local"
                          value={field.state.value ?? ''}
                        />
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />
                <FormField
                  name="updated_at_end"
                  render={(field) => (
                    <FormItem>
                      <FormLabel>Updated Before</FormLabel>
                      <FormControl>
                        <Input
                          name={field.name}
                          onBlur={field.handleBlur}
                          onChange={ev => field.handleChange(ev.target.valueAsDate)}
                          type="datetime-local"
                          value={field.state.value ?? ''}
                        />
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />
                <FormField
                  name="last_modified_at_start"
                  render={(field) => (
                    <FormItem>
                      <FormLabel>Last Modified After</FormLabel>
                      <FormControl>
                        <Input
                          name={field.name}
                          onBlur={field.handleBlur}
                          onChange={ev => field.handleChange(ev.target.valueAsDate)}
                          type="datetime-local"
                          value={field.state.value ?? ''}
                        />
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />
                <FormField
                  name="last_modified_at_end"
                  render={(field) => (
                    <FormItem>
                      <FormLabel>Last Modified Before</FormLabel>
                      <FormControl>
                        <Input
                          name={field.name}
                          onBlur={field.handleBlur}
                          onChange={ev => field.handleChange(ev.target.valueAsDate)}
                          type="datetime-local"
                          value={field.state.value ?? ''}
                        />
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />
              </div>
            </CollapsibleContent>
          </Collapsible>

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