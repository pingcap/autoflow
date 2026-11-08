import { handleResponse } from '@/lib/request';
import { z, type ZodType } from 'zod';

export const enum VerifyStatus {
  CREATED = 'CREATED',
  EXTRACTING = 'EXTRACTING',
  VALIDATING = 'VALIDATING',
  SUCCESS = 'SUCCESS',
  FAILED = 'FAILED',
  SKIPPED = 'SKIPPED'
}

export interface MessageVerifyResponse {
  status: VerifyStatus;
  message?: string | null;
  runs_report: string | null;
}

export namespace MessageVerifyResponse {
  export type Run = {
    sql: string
    explanation: string
    success: boolean
    results?: any[][]
    sql_error_code?: number | null
    sql_error_message?: string | null
    llm_verification?: string | null
    warnings?: string[]
  }
}

const getVerifyResponse = z.object({
  status: z.enum([VerifyStatus.CREATED, VerifyStatus.EXTRACTING, VerifyStatus.VALIDATING, VerifyStatus.SUCCESS, VerifyStatus.FAILED, VerifyStatus.SKIPPED]),
  message: z.string().nullish(),
  runs_report: z.string().nullable(),
}) satisfies ZodType<MessageVerifyResponse, any>;

export async function getVerify (url: string) {
  return await fetch(url).then(handleResponse(getVerifyResponse));
}

export function isFinalVerifyState (state: VerifyStatus) {
  return [VerifyStatus.SUCCESS, VerifyStatus.FAILED, VerifyStatus.SKIPPED].includes(state);
}

export function isVisibleVerifyState (state: VerifyStatus) {
  return [VerifyStatus.SUCCESS, VerifyStatus.FAILED].includes(state);
}

function assertEnabled (value: string | undefined): asserts value is string {
  if (!value) {
    throw new Error('Experimental message verify not enabled.');
  }
}
