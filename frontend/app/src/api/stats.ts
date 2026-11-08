import { authenticationHeaders, handleResponse, requestUrl } from '@/lib/request';
import { zodDateOnlyString } from '@/lib/zod';
import { format } from 'date-fns';
import { z, type ZodType } from 'zod';

export interface TrendBaseItem {
  date: Date;
}

export interface ChatUserTrendItem extends TrendBaseItem {
  user: number;
  anonymous: number;
}

export type ChatOriginTrendItem = TrendBaseItem & Omit<{
  [key: string]: number
}, 'date'>

export interface TrendResponse<T> {
  start_date: Date;
  end_date: Date;
  values: T[];
}

const trendBaseSchema = z.object({
  date: zodDateOnlyString(),
});

const chatUserTrendItemSchema = trendBaseSchema.extend({
  user: z.number(),
  anonymous: z.number(),
}) satisfies ZodType<ChatUserTrendItem, any>;

const chatOriginTrendItemSchema = trendBaseSchema.and(
  z.preprocess((input) => {
    if (input && typeof input === 'object') {
      const { date: _, ...ret } = input as any;
      return ret;
    } else {
      return input;
    }
  }, z.record(z.string(), z.number())),
) satisfies ZodType<ChatOriginTrendItem, any>;

function trendResponse<T> (item: ZodType<T, any>): ZodType<TrendResponse<T>, any> {
  return z.object({
    start_date: zodDateOnlyString(),
    end_date: zodDateOnlyString(),
    values: item.array(),
  });
}

function trendParams (start: Date, end: Date) {
  return {
    start_date: format(start, 'yyyy-MM-dd'),
    end_date: format(end, 'yyyy-MM-dd'),
  };
}

export async function getChatUserTrend (startDate: Date, endDate: Date) {
  return await fetch(requestUrl('/api/v1/admin/stats/trend/chat-user', trendParams(startDate, endDate)), {
    headers: {
      ...await authenticationHeaders(),
    },
  })
    .then(handleResponse(trendResponse(chatUserTrendItemSchema)));
}

export async function getChatOriginTrend (startDate: Date, endDate: Date) {
  return await fetch(requestUrl('/api/v1/admin/stats/trend/chat-origin', trendParams(startDate, endDate)), {
    headers: {
      ...await authenticationHeaders(),
    },
  })
    .then(handleResponse(trendResponse(chatOriginTrendItemSchema)));
}
