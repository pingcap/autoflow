import { cn } from '@/lib/utils';
import type { ReactNode } from 'react';

export function OptionDetail ({
  valueClassName,
  title,
  value,
}: {
  valueClassName?: string
  title: string
  value: ReactNode
}) {
  return (
    <div className="flex items-center gap-2">
      <dt className="text-muted-foreground text-xs">{title}</dt>
      <dd className={cn('font-medium ml-auto', valueClassName)}>{value}</dd>
    </div>
  );
}
