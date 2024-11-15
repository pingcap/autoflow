'use client';

import Link from 'next/link';

export function DatasourceCell ({ id, name }: { id: number, name: string }) {
  return <Link className="underline" href={`/datasources/${id}`}>{name}</Link>;
}

export function KnowledgeBaseCell ({ id, name }: { id?: number, name?: string }) {
  if (id == null) {
    return <span className="text-muted-foreground">-</span>;
  }
  return <Link className="underline" href={`/knowledge-bases/${id}`}>{name ?? 'Unnamed'}</Link>;
}
