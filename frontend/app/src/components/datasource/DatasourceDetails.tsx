'use client';

import { useDatasource } from '@/components/datasource/hooks';
import { DateFormat } from '@/components/date-format';
import { OptionDetail } from '@/components/option-detail';
import { Badge } from '@/components/ui/badge';

export function DatasourceDetails ({ id }: { id: number }) {
  return (
    <>
      <DatasourceFields id={id} />
      <DatasourceUploadFiles id={id} />
    </>
  );
}

function DatasourceFields ({ id }: { id: number }) {
  const { datasource } = useDatasource(id);
  return (
    <div className="space-y-2 text-sm rounded p-4 border">
      <OptionDetail title="ID" value={id} />
      <OptionDetail title="Type" value={datasource?.data_source_type} />
      <OptionDetail title="Created At" value={<DateFormat date={datasource?.created_at} />} />
      <OptionDetail title="Updated At" value={<DateFormat date={datasource?.updated_at} />} />
      {(datasource?.data_source_type === 'web_sitemap') && (
        <OptionDetail title="URL" value={datasource?.config.url} />
      )}
      {(datasource?.data_source_type === 'web_single_page') && (
        <OptionDetail title="URL" value={<ul>{datasource?.config.urls.map(url => <li key={url}><a className="underline" href={url} target="_blank">{url}</a></li>)}</ul>} />
      )}
      <OptionDetail title="Name" value={datasource?.name} />
    </div>
  );
}

function DatasourceUploadFiles ({ id }: { id: number }) {
  const { datasource } = useDatasource(id);

  if (datasource?.data_source_type !== 'file' || datasource.config.length === 0) {
    return null;
  }
  return (
    <section className="space-y-4">
      <h3 className="font-medium">Files</h3>
      <div className="flex gap-2 flex-wrap">
        {datasource.config.map(file => (
          <Badge key={file.file_id} variant="secondary" className="gap-1">
            <span>
              {file.file_name}
            </span>
            <span className="font-normal text-muted-foreground">#{file.file_id}</span>
          </Badge>
        ))}
      </div>
    </section>
  );
}
