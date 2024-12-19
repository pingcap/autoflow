'use client';;
import { use } from "react";

import { getEmbeddingModel } from '@/api/embedding-models';
import { AdminPageHeading } from '@/components/admin-page-heading';
import { ConfigViewer } from '@/components/config-viewer';
import { DateFormat } from '@/components/date-format';
import { OptionDetail } from '@/components/option-detail';
import { Loader2Icon } from 'lucide-react';
import useSWR from 'swr';

export default function Page(props: { params: Promise<{ id: string }> }) {
  const params = use(props.params);
  const { data } = useSWR(`api.embedding-models.get?id=${params.id}`, () => getEmbeddingModel(parseInt(params.id)));

  return (
    <>
      <AdminPageHeading
        breadcrumbs={[
          { title: 'Models' },
          { title: 'Embedding Models', url: '/embedding-models', docsUrl: '/docs/embedding-model' },
          { title: data ? data.name : <Loader2Icon className="size-4 animate-spin repeat-infinite" /> },
        ]}
      />
      <div className="max-w-screen-sm space-y-4">
        <div className="space-y-2 text-sm rounded p-4 border">
          <OptionDetail title="ID" value={data?.id} />
          <OptionDetail title="Name" value={data?.name} />
          <OptionDetail title="Provider" value={data?.provider} />
          <OptionDetail title="Model" value={data?.model} />
          <OptionDetail title="Vector Dimentions" value={data?.vector_dimension} />
          <OptionDetail title="Config" value={data?.config && <ConfigViewer value={data.config}></ConfigViewer>} />
          <OptionDetail title="Is Default" value={data?.is_default ? 'Yes' : 'No'} valueClassName={data?.is_default ? 'text-success' : 'text-muted-foreground'} />
          <OptionDetail title="Created At" value={<DateFormat date={data?.created_at} />} />
          <OptionDetail title="Updated At" value={<DateFormat date={data?.updated_at} />} />
        </div>
      </div>
    </>
  );
}
