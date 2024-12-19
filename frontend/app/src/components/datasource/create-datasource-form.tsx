import { type BaseCreateDatasourceParams, createDatasource, type CreateDatasourceSpecParams, uploadFiles } from '@/api/datasources';
import { FormInput } from '@/components/form/control-widget';
import { FormFieldBasicLayout, FormPrimitiveArrayFieldBasicLayout } from '@/components/form/field-layout.beta';
import { onSubmitHelper } from '@/components/form/utils';
import { FilesInput } from '@/components/form/widgets/FilesInput';
import { Button } from '@/components/ui/button';
import { Form, formDomEventHandlers, FormField, FormItem, FormLabel, useFormContext } from '@/components/ui/form.beta';
import { ToggleGroup, ToggleGroupItem } from '@/components/ui/toggle-group';
import { zodFile } from '@/lib/zod';
import { useForm as useTanstackForm } from '@tanstack/react-form';
import { FileDownIcon, GlobeIcon, PaperclipIcon } from 'lucide-react';
import { useSearchParams } from 'next/navigation';
import { z } from 'zod';

const types = ['file', 'web_single_page', 'web_sitemap'] as const;

const isType = (value: string | null): value is typeof types[number] => types.includes(value as any);

export function CreateDatasourceForm ({ knowledgeBaseId, transitioning, onCreated }: { knowledgeBaseId: number, transitioning?: boolean, onCreated?: () => void }) {
  const usp = useSearchParams()!;
  const uType = usp.get('type');

  const form = useTanstackForm<CreateDatasourceFormParams>({
    validators: {
      onSubmit: createDatasourceSchema,
    },
    defaultValues: switchDatasource({
      data_source_type: 'file',
      name: '',
      files: [],
    }, isType(uType) ? uType : 'file'),
    onSubmit: onSubmitHelper(createDatasourceSchema, async (data) => {
      const createParams = await preCreate(data);
      await createDatasource(knowledgeBaseId, createParams);
      onCreated?.();
    }),
  });

  return (
    <Form form={form} disabled={transitioning}>
      <form className="max-w-screen-sm space-y-4" {...formDomEventHandlers(form, transitioning)}>
        <DataSourceTypeField />
        <DataSourceTypeSpecFields />
        <FormFieldBasicLayout name="name" label="Datasource Name">
          <FormInput />
        </FormFieldBasicLayout>
        <Button type="submit" disabled={form.state.isSubmitting}>
          Create
        </Button>
      </form>
    </Form>
  );
}

function DataSourceTypeField () {
  return (
    <FormField<CreateDatasourceFormParams, 'data_source_type'>
      name="data_source_type"
      render={(field, form) => (
        <FormItem>
          <FormLabel>
            Data Source Type
          </FormLabel>
          <ToggleGroup
            className="w-max"
            type="single"
            value={field.state.value}
            onValueChange={(value => {
              form.reset(switchDatasource(form.state.values, value as never));
            })}
            onBlur={field.handleBlur}
          >
            <ToggleGroupItem value="file">
              <PaperclipIcon className="size-4 mr-2" />
              File
            </ToggleGroupItem>
            <ToggleGroupItem value="web_single_page">
              <FileDownIcon className="size-4 mr-2" />
              Web Single Page
            </ToggleGroupItem>
            <ToggleGroupItem value="web_sitemap">
              <GlobeIcon className="size-4 mr-2" />
              Web Sitemap
            </ToggleGroupItem>
          </ToggleGroup>
        </FormItem>
      )}
    />
  );
}

function DataSourceTypeSpecFields () {
  const { form } = useFormContext<CreateDatasourceFormParams>();

  return (
    <form.Subscribe selector={state => state.values.data_source_type}>
      {(type) => (
        <>
          {type === 'file' && (
            <FormFieldBasicLayout name="files" label="Files" description="Currently support Markdown (*.md), PDF (*.pdf), Microsoft Word (*.docx), Microsoft PowerPoint (*.pptx), Microsoft Excel (*.xlsx) and Text (*.txt) files.">
              <FilesInput accept={['text/plain', 'application/pdf', '.md', '.docx', '.pptx', '.xlsx']} />
            </FormFieldBasicLayout>
          )}
          {type === 'web_single_page' && (
            <FormPrimitiveArrayFieldBasicLayout name="urls" label="Page URL" defaultValue={() => ''}>
              <FormInput placeholder="https://example.com/" />
            </FormPrimitiveArrayFieldBasicLayout>
          )}
          {type === 'web_sitemap' && (
            <FormFieldBasicLayout name="url" label="Sitemap URL">
              <FormInput placeholder="https://example.com/sitemap.xml" />
            </FormFieldBasicLayout>
          )}
        </>
      )}
    </form.Subscribe>
  );
}

export type CreateDatasourceFormParams = z.infer<typeof createDatasourceSchema>;

export const createDatasourceSchema = z.object({
  name: z.string().trim().min(1, 'Must not blank'),
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

function switchDatasource (data: CreateDatasourceFormParams, type: CreateDatasourceSpecParams['data_source_type']): CreateDatasourceFormParams {
  if (data.data_source_type === type) {
    return data;
  }

  switch (type) {
    case 'file':
      return {
        name: data.name,
        data_source_type: 'file',
        files: [],
      };
    case 'web_single_page':
      return {
        name: data.name,
        data_source_type: 'web_single_page',
        urls: [],
      };
    case 'web_sitemap':
      return {
        name: data.name,
        data_source_type: 'web_sitemap',
        url: '',
      };
  }
}

async function preCreate (ds: CreateDatasourceFormParams): Promise<BaseCreateDatasourceParams & CreateDatasourceSpecParams> {
  switch (ds.data_source_type) {
    case 'file': {
      const { files, ...rest } = ds;
      const uploadedFiles = await uploadFiles(ds.files);
      return {
        ...rest,
        config: uploadedFiles.map(f => ({
          file_id: f.id,
          file_name: f.name,
        })),
      };
    }
    case 'web_single_page': {
      const { urls, ...rest } = ds;

      return {
        ...rest,
        config: { urls },
      };
    }

    case 'web_sitemap':
      const { url, ...rest } = ds;

      return {
        ...rest,
        config: { url },
      };
  }
}
