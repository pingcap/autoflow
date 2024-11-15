import { AdminPageHeading } from '@/components/admin-page-heading';
import { DatasourceDeprecationAlert } from '@/components/datasource/DatasourceDeprecationAlert';
import { DatasourceDetails } from '@/components/datasource/DatasourceDetails';
import { DatasourceName } from '@/components/datasource/DatasourceName';

export default function DatasourcePage ({ params }: { params: { id: string } }) {
  const id = parseInt(params.id);

  return (
    <div className="max-w-screen-md space-y-8">
      <AdminPageHeading
        breadcrumbs={[
          { title: 'Datasources', url: '/datasources' },
          { title: <DatasourceName id={id} />, url: `/datasources/${id}` },
        ]}
      />
      <DatasourceDeprecationAlert />
      <DatasourceDetails id={id} />
    </div>
  );
}