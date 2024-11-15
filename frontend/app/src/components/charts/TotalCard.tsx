import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Skeleton } from '@/components/ui/skeleton';
import { cn } from '@/lib/utils';
import type { ReactNode } from 'react';

const nf = new Intl.NumberFormat('en-US', {});

export interface TotalCardProps {
  isLoading: boolean;
  title: string,
  icon: ReactNode,
  total?: number | null | undefined,
  children?: ReactNode
}

export function TotalCard ({ isLoading = false, title, icon, total, children }: TotalCardProps) {
  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle className="text-sm font-medium">{title}</CardTitle>
        {icon}
      </CardHeader>
      <CardContent>
        <div className={cn('text-2xl font-bold', total == null && 'h-8 pt-2')}>
          {isLoading ? <Skeleton className="w-12 h-4" /> : nf.format(total || 0)}
        </div>
        <p className="text-xs text-muted-foreground mt-4">
          {children}
        </p>
      </CardContent>
    </Card>
  );
}
