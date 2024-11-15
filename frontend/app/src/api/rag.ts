import { z, type ZodType } from 'zod';

export const indexStatuses = [
  'not_started',
  'pending',
  'running',
  'completed',
  'failed',
] as const;

export type IndexStatus = typeof indexStatuses[number];

export type IndexProgress = Partial<Record<IndexStatus, number>>

export type IndexTotalStats = {
  total: number
}

export const indexStatusSchema = z.enum(indexStatuses) satisfies ZodType<IndexStatus>;

export const totalSchema = z.object({
  total: z.number(),
}) satisfies ZodType<IndexTotalStats>;

export const indexSchema = z.object({
  not_started: z.number().optional(),
  pending: z.number().optional(),
  running: z.number().optional(),
  completed: z.number().optional(),
  failed: z.number().optional(),
}) satisfies ZodType<IndexProgress>;
