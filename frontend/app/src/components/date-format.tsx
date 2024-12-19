'use client';

import { format } from 'date-fns';

export function DateFormat ({ className, date, format: formatStr = 'yyyy-MM-dd HH:mm:ss' }: { className?: string, date: Date | null | undefined, format?: string }) {
  return (
    <time className={className}>
      {date ? isNaN(date.getTime()) ? 'Invalid Date' : format(date, formatStr) : '-'}
    </time>
  );
}
