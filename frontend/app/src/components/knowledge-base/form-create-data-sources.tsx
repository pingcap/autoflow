import type { BaseCreateDatasourceParams, CreateDatasourceSpecParams } from '@/api/datasources';
import type { CreateKnowledgeBaseParams } from '@/api/knowledge-base';
import { FormInput } from '@/components/form/control-widget';
import { FormFieldBasicLayout, FormPrimitiveArrayFieldBasicLayout } from '@/components/form/field-layout';
import { FilesInput } from '@/components/form/widgets/FilesInput';
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from '@/components/ui/accordion';
import { Button } from '@/components/ui/button';
import { FormItem, FormLabel, FormMessage } from '@/components/ui/form';
import { FormArrayField } from '@/components/ui/form.ext';
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { cn } from '@/lib/utils';
import { zodFile } from '@/lib/zod';
import { PlusIcon } from 'lucide-react';
import { useFormContext } from 'react-hook-form';
import { z } from 'zod';

export const createDatasourceSchema = z.object({
  name: z.string().trim().min(1, 'Must not blank'),
  description: z.string(),
}).and(z.discriminatedUnion('data_source_type', [
  z.object({
    data_source_type: z.literal('file'),
    files: zodFile().array().min(1),
  }),
  z.object({
    data_source_type: z.literal('web_single_page'),
    urls: z.string().url().array().min(1),
  }),
  z.object({
    data_source_type: z.literal('web_sitemap'),
    url: z.string().url(),
  }),
]));

export function FormCreateDataSources () {
  return (
    <FormArrayField<CreateKnowledgeBaseParams, 'data_sources'>
      name="data_sources"
      render={({ field: { fields, update, append, remove } }) => (
        <FormItem>
          <FormLabel className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70">Datasource</FormLabel>
          <Accordion className="border rounded px-4" type="multiple" defaultValue={[fields[0]?.id ?? '']}>
            {(fields).map((field, index) => (
              <AccordionItem key={field.id} value={field.id} className="space-y-4 last-of-type:border-b-0">
                <AccordionTrigger className="hover:no-underline">
                  <DatasourceName index={index} />
                </AccordionTrigger>
                <AccordionContent className="space-y-4 px-2">
                  <FormFieldBasicLayout name={`data_sources.${index}.name`} label="Datasource Name">
                    <FormInput />
                  </FormFieldBasicLayout>
                  <Tabs
                    value={field.data_source_type}
                    onValueChange={(value => update(index, { ...switchDatasource(field, value as never) }))}
                  >
                    <TabsList>
                      <TabsTrigger value="file">
                        File
                      </TabsTrigger>
                      <TabsTrigger value="web_single_page">
                        Web Single Page
                      </TabsTrigger>
                      <TabsTrigger value="web_sitemap">
                        Web Sitemap
                      </TabsTrigger>
                    </TabsList>
                  </Tabs>
                  {field.data_source_type === 'file' && (
                    <FormFieldBasicLayout name={`data_sources.${index}.files`} label="Files" description="Currently support Markdown (*.md), PDF (*.pdf), Microsoft Word (*.docx), Microsoft PowerPoint (*.pptx), Microsoft Excel (*.xlsx) and Text (*.txt) files.">
                      <FilesInput accept={['text/plain', 'application/pdf', '.md', '.docx', '.pptx', '.xlsx']} />
                    </FormFieldBasicLayout>
                  )}
                  {field.data_source_type === 'web_single_page' && (
                    <FormPrimitiveArrayFieldBasicLayout name={`data_sources.${index}.urls`} label="Page URL" defaultValue={() => ''}>
                      <FormInput placeholder="https://example.com/" />
                    </FormPrimitiveArrayFieldBasicLayout>
                  )}
                  {field.data_source_type === 'web_sitemap' && (
                    <FormFieldBasicLayout name={`data_sources.${index}.url`} label="Sitemap URL">
                      <FormInput placeholder="https://example.com/sitemap.xml" />
                    </FormFieldBasicLayout>
                  )}
                  <FormMessage />
                  {fields.length > 1 && <Button type="button" variant="outline" onClick={() => remove(index)}>Remove</Button>}
                </AccordionContent>
              </AccordionItem>
            ))}
          </Accordion>
          <FormMessage />
          <Button
            type="button"
            variant="outline"
            onClick={() => append({ name: '', data_source_type: 'file', config: [] })}
          >
            <PlusIcon className="size-4 ml-1" />
            New Datasource
          </Button>
        </FormItem>
      )}
    />)
    ;
}

function DatasourceName ({ index }: { index: number }) {
  const { watch, formState } = useFormContext<CreateKnowledgeBaseParams>();
  const errors = formState.errors.data_sources?.[index];
  const name = watch(`data_sources.${index}.name`);

  return (
    <span className={cn(!!errors && 'text-destructive')}>
      <span className="text-muted-foreground mr-1">{index + 1}.</span>
      {name || <span className={cn(!errors && 'text-muted-foreground')}>Unnamed</span>}
    </span>
  );
}

function switchDatasource (data: BaseCreateDatasourceParams & CreateDatasourceSpecParams, type: CreateDatasourceSpecParams['data_source_type']): BaseCreateDatasourceParams & CreateDatasourceSpecParams {
  switch (type) {
    case 'file':
      return {
        name: data.name,
        data_source_type: 'file',
        config: [],
      };
    case 'web_single_page':
      return {
        name: data.name,
        data_source_type: 'web_single_page',
        config: { urls: [] },
      };
    case 'web_sitemap':
      return {
        name: data.name,
        data_source_type: 'web_sitemap',
        config: { url: '' },
      };
  }
}
