'use client';;
import { use } from "react";

import { DatasourceCard } from '@/components/datasource/datasource-card';
import { DatasourceCreateOption } from '@/components/datasource/datasource-create-option';
import { NoDatasourcePlaceholder } from '@/components/datasource/no-datasource-placeholder';
import { useAllKnowledgeBaseDataSources } from '@/components/knowledge-base/hooks';
import { Skeleton } from '@/components/ui/skeleton';
import { FileDownIcon, GlobeIcon, PaperclipIcon } from 'lucide-react';

export default function KnowledgeBaseDataSourcesPage(props: { params: Promise<{ id: string }> }) {
  const params = use(props.params);
  const id = parseInt(decodeURIComponent(params.id));
  const { data: dataSources, isLoading } = useAllKnowledgeBaseDataSources(id);

  return (
    <div className="space-y-8 max-w-screen-sm">
      <section className="space-y-4">
        <h3>Create Data Source</h3>
        <div className="grid md:grid-cols-3 gap-4">
          <DatasourceCreateOption
            knowledgeBaseId={id}
            type="file"
            icon={<PaperclipIcon className="size-4 flex-shrink-0" />}
            title="Files"
          >
            Upload files
          </DatasourceCreateOption>
          <DatasourceCreateOption
            knowledgeBaseId={id}
            type="web_single_page"
            icon={<FileDownIcon className="size-4 flex-shrink-0" />}
            title="Web Pages"
          >
            Select pages.
          </DatasourceCreateOption>
          <DatasourceCreateOption
            knowledgeBaseId={id}
            type="web_sitemap"
            icon={<GlobeIcon className="size-4 flex-shrink-0" />}
            title="Website by sitemap"
          >
            Select web sitemap.
          </DatasourceCreateOption>
        </div>
      </section>
      <section className="space-y-4">
        <h3>Browse existing Data Sources</h3>
        {isLoading && <Skeleton className="h-20 rounded-lg" />}
        {dataSources?.map(datasource => (
          <DatasourceCard key={datasource.id} knowledgeBaseId={id} datasource={datasource} />
        ))}
        {dataSources?.length === 0 && (
          <NoDatasourcePlaceholder />
        )}
      </section>
    </div>
  );
}
