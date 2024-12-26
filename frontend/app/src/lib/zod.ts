import { parse, parseJSON, startOfToday } from 'date-fns';
import { z, ZodType } from 'zod';

const BASE_DATE = startOfToday();

export function zodDateOnlyString () {
  return z.string().regex(/^\d\d\d\d-\d\d-\d\d$/).transform(str => {
    return parse(str, 'yyyy-MM-dd', BASE_DATE);
  });
}

export function zodJsonDate (message?: string) {
  return z.string().pipe(d);
}

export function zodJsonText () {
  return z.string()
    .refine(val => {
      try {
        JSON.parse(val);
        return true;
      } catch {
        return false;
      }
    }, { message: 'Invalid JSON' })
    .transform(value => JSON.parse(value));
}

export function zodFile () {
  return z.custom<File>(value => value instanceof File);
}

export function zodJson () {
  return z.union([
    z.string(),
    z.number(),
    z.boolean(),
    z.any().array(),
    z.object({}).passthrough(),
  ]);
}

const d = z
  .custom<string>(data => {
    if (typeof data !== 'string') {
      throw new Error('Requires a date string');
    }

    const date = parseJSON(data);
    return !isNaN(date.getTime());
  }, { message: 'Invalid date' })
  .transform(out => {
    return parseJSON(out);
  });

export interface PageParams {
  page?: number; // 1 based
  size?: number;
}

export interface Page<T> {
  items: T[];
  total: number;
  page: number;
  size: number;
  pages: number;
}

export function zodPage<Z extends ZodType> (itemSchema: Z) {
  return z.object({
    items: itemSchema.array(),
    total: z.number(),
    page: z.number(),
    size: z.number(),
    pages: z.number(),
  });
}

export function noPage<T> (data: T[]): Page<T> {
  return {
    items: data,
    total: data.length,
    page: 1,
    size: data.length,
    pages: 1,
  };
}