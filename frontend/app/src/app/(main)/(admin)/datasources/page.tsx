import { AdminPageHeading } from '@/components/admin-page-heading';
import { DatasourceDeprecationAlert } from '@/components/datasource/DatasourceDeprecationAlert';
import { DatasourceTable } from '@/components/datasource/DatasourceTable';

export default function ChatEnginesPage () {
  return (
    <>
      <AdminPageHeading title="Datasources" />
      <DatasourceDeprecationAlert />
      <DatasourceTable />
    </>
  );
}
