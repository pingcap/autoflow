import type { DeprecatedBaseCreateDatasourceParams } from '@/api/datasources';
import { z, type ZodType } from 'zod';

export const createDatasourceBaseSchema = z.object({
  name: z.string().trim().min(1, 'Must not blank'),
  description: z.string(),
  build_kg_index: z.boolean(),
  llm_id: z.number().nullable(),
}) satisfies ZodType<DeprecatedBaseCreateDatasourceParams, any, any>;
