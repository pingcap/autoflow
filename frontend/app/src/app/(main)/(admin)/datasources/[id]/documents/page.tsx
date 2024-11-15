import { getDatasource } from '@/api/datasources';
import { AdminPageHeading } from '@/components/admin-page-heading';
import { DatasourceDeprecationAlert } from '@/components/datasource/DatasourceDeprecationAlert';
import { DocumentsTable } from '@/components/documents/documents-table';
import { isServerError } from '@/lib/request';
import { notFound } from 'next/navigation';
import { cache } from 'react';

export default async function ChatEnginesPage ({ params }: { params: { id: string } }) {
  const datasource = await cachedGetDatasource(parseInt(params.id));

  return (
    <>
      <AdminPageHeading
        breadcrumbs={[
          { title: 'Datasources', url: '/datasources' },
          { title: datasource.name, url: `/datasources/${datasource.id}` },
          { title: 'Documents' },
        ]}
      />
      <DatasourceDeprecationAlert />
      <DocumentsTable knowledgeBaseId={datasource.id} />
    </>
  );
}

const cachedGetDatasource = cache(async (id: number) => {
  try {
    return await getDatasource(id);
  } catch (error) {
    if (isServerError(error, [404])) {
      notFound();
    } else {
      return Promise.reject(error);
    }
  }
});

