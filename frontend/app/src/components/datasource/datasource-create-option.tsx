'use client';

import { NextLink } from '@/components/nextjs/NextLink';
import type { ReactNode } from 'react';

export function DatasourceCreateOption ({
  knowledgeBaseId,
  type,
  icon,
  title,
  children,
}: {
  knowledgeBaseId: number
  type: string
  icon?: ReactNode
  title: ReactNode
  children?: ReactNode
}) {
  return (
    <NextLink
      href={`/knowledge-bases/${knowledgeBaseId}/data-sources/new?type=${type}`}
      className="block space-y-2 h-auto"
      variant="secondary"
    >
      <div className="break-words text-wrap">
        <span className="inline-flex items-center h-max mr-1 align-middle">
          {icon}
        </span>
        {title}
      </div>
      <div className="text-muted-foreground text-xs font-normal break-words text-wrap">
        {children}
      </div>
    </NextLink>
  );
}
