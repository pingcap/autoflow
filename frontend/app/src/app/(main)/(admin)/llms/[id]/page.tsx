'use client';

import { deleteLlm, getLlm } from '@/api/llms';
import { AdminPageHeading } from '@/components/admin-page-heading';
import { DangerousActionButton } from '@/components/dangerous-action-button';
import { DateFormat } from '@/components/date-format';
import { ConfigViewer } from '@/components/config-viewer';
import { OptionDetail } from '@/components/option-detail';
import { Loader2Icon } from 'lucide-react';
import { useRouter } from 'next/navigation';
import { useTransition, use } from 'react';
import useSWR from 'swr';

export default function Page(props: { params: Promise<{ id: string }> }) {
  const params = use(props.params);
  const router = useRouter();
  const { data } = useSWR(`api.llms.get?id=${params.id}`, () => getLlm(parseInt(params.id)));
  const [transitioning, startTransition] = useTransition();

  return (
    <>
      <AdminPageHeading
        breadcrumbs={[
          { title: 'Models' },
          { title: 'LLMs', url: '/llms', docsUrl: '/docs/llm' },
          { title: data ? data.name : <Loader2Icon className="size-4 animate-spin repeat-infinite" /> },
        ]}
      />
      <div className="max-w-screen-sm space-y-4">
        <div className="space-y-2 text-sm rounded p-4 border">
          <OptionDetail title="ID" value={data?.id} />
          <OptionDetail title="Name" value={data?.name} />
          <OptionDetail title="Provider" value={data?.provider} />
          <OptionDetail title="Model" value={data?.model} />
          <OptionDetail title="Config" value={data?.config && <ConfigViewer value={data.config}></ConfigViewer>} />
          <OptionDetail title="Is Default" value={data?.is_default ? 'Yes' : 'No'} valueClassName={data?.is_default ? 'text-success' : 'text-muted-foreground'} />
          <OptionDetail title="Created At" value={<DateFormat date={data?.created_at} />} />
          <OptionDetail title="Updated At" value={<DateFormat date={data?.updated_at} />} />
        </div>
        <div>
          <DangerousActionButton
            variant="destructive"
            disabled={transitioning}
            action={async () => {
              await deleteLlm(parseInt(params.id));
              startTransition(() => {
                router.push('/llms');
              });
            }}
          >
            Delete
          </DangerousActionButton>
        </div>
      </div>
    </>
  );
}
