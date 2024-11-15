import { type IndexProgress, indexSchema, type IndexTotalStats, totalSchema } from '@/api/rag';
import { authenticationHeaders, handleErrors, handleResponse, type Page, type PageParams, requestUrl, zodPage } from '@/lib/request';
import { zodJsonDate } from '@/lib/zod';
import { z, type ZodType } from 'zod';

export interface DatasourceBase {
  id: number;
  name: string;
}

interface DeprecatedDatasourceBase extends DatasourceBase {
  created_at: Date;
  updated_at: Date;
  user_id: string | null;
  build_kg_index: boolean;
  llm_id: number | null;
}

export type DeprecatedDatasource = DeprecatedDatasourceBase & DatasourceSpec

type DatasourceSpec = ({
  data_source_type: 'file'
  config: { file_id: number, file_name: string }[]
} | {
  data_source_type: 'web_sitemap'
  config: { url: string }
} | {
  data_source_type: 'web_single_page'
  config: { urls: string[] }
})

export type Datasource = DatasourceBase & DatasourceSpec;

export type DataSourceIndexProgress = {
  vector_index: IndexProgress
  documents: IndexTotalStats
  chunks: IndexTotalStats
  kg_index?: IndexProgress
  relationships?: IndexTotalStats
}

export interface BaseCreateDatasourceParams {
  name: string;
}

export interface DeprecatedBaseCreateDatasourceParams extends BaseCreateDatasourceParams {
  description: string;
  /**
   * @deprecated
   */
  build_kg_index: boolean;
  /**
   * @deprecated
   */
  llm_id: number | null;
}

export type CreateDatasourceSpecParams = ({
  data_source_type: 'file'
  config: { file_id: number, file_name: string }[]
} | {
  data_source_type: 'web_single_page'
  config: { urls: string[] }
} | {
  data_source_type: 'web_sitemap'
  config: { url: string }
});

export type CreateDatasourceParams = DeprecatedBaseCreateDatasourceParams & CreateDatasourceSpecParams;

export interface Upload {
  created_at?: Date;
  updated_at?: Date;
  id: number;
  name: string;
  size: number;
  path: string;
  mime_type: string;
  user_id: string;
}

export type DatasourceVectorIndexError = {
  document_id: number
  document_name: string
  source_uri: string
  error: string | null
}

export type DatasourceKgIndexError = {
  chunk_id: string
  source_uri: string
  error: string | null
}

const deprecatedBaseDatasourceSchema = z.object({
  id: z.number(),
  name: z.string(),
  created_at: zodJsonDate(),
  updated_at: zodJsonDate(),
  user_id: z.string().nullable(),
  build_kg_index: z.boolean(),
  llm_id: z.number().nullable(),
});

const datasourceSpecSchema = z.discriminatedUnion('data_source_type', [
  z.object({
    data_source_type: z.literal('file'),
    config: z.array(z.object({ file_id: z.number(), file_name: z.string() })),
  }),
  z.object({
    data_source_type: z.enum(['web_single_page']),
    config: z.object({ urls: z.string().array() }).or(z.object({ url: z.string() })).transform(obj => {
      if ('url' in obj) {
        return { urls: [obj.url] };
      } else {
        return obj;
      }
    }),
  }),
  z.object({
    data_source_type: z.enum(['web_sitemap']),
    config: z.object({ url: z.string() }),
  })],
) satisfies ZodType<DatasourceSpec, any, any>;

export const deprecatedDatasourceSchema = deprecatedBaseDatasourceSchema
  .and(datasourceSpecSchema) satisfies ZodType<DeprecatedDatasource, any, any>;

export const datasourceSchema = z.object({
  id: z.number(),
  name: z.string(),
}).and(datasourceSpecSchema) satisfies ZodType<Datasource, any, any>;

const uploadSchema = z.object({
  id: z.number(),
  name: z.string(),
  size: z.number(),
  path: z.string(),
  mime_type: z.string(),
  user_id: z.string(),
  created_at: zodJsonDate().optional(),
  updated_at: zodJsonDate().optional(),
}) satisfies ZodType<Upload, any, any>;

const datasourceOverviewSchema = z.object({
  vector_index: indexSchema,
  documents: totalSchema,
  chunks: totalSchema,
  kg_index: indexSchema.optional(),
  relationships: totalSchema.optional(),
}) satisfies ZodType<DataSourceIndexProgress>;

export async function listDataSources ({ page = 1, size = 10 }: PageParams = {}): Promise<Page<DeprecatedDatasource>> {
  return fetch(requestUrl('/api/v1/admin/datasources', { page, size }), {
    headers: await authenticationHeaders(),
  }).then(handleResponse(zodPage(deprecatedDatasourceSchema)));
}

export async function getDatasource (id: number): Promise<DeprecatedDatasource> {
  return fetch(requestUrl(`/api/v1/admin/datasources/${id}`), {
    headers: await authenticationHeaders(),
  }).then(handleResponse(deprecatedDatasourceSchema));
}

export async function deleteDatasource (id: number): Promise<void> {
  await fetch(requestUrl(`/api/v1/admin/datasources/${id}`), {
    method: 'DELETE',
    headers: await authenticationHeaders(),
  }).then(handleErrors);
}

export async function createDatasource (params: CreateDatasourceParams) {
  return fetch(requestUrl(`/api/v1/admin/datasources`), {
    method: 'POST',
    headers: {
      ...await authenticationHeaders(),
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(params),
  }).then(handleResponse(deprecatedDatasourceSchema));
}

export async function uploadFiles (files: File[]) {
  const formData = new FormData();
  files.forEach((file) => {
    formData.append('files', file);
  });

  return fetch(requestUrl(`/api/v1/admin/uploads`), {
    method: 'POST',
    headers: {
      ...await authenticationHeaders(),
    },
    body: formData,
  }).then(handleResponse(uploadSchema.array()));
}
