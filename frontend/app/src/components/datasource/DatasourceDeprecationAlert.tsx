import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { AlertCircleIcon } from 'lucide-react';

export function DatasourceDeprecationAlert () {
  return (
    <Alert variant="warning">
      <AlertCircleIcon />
      <AlertTitle>
        Bare Datasource management was deprecated.
      </AlertTitle>
      <AlertDescription>
        TiDB.ai now uses Knowledge Base to manage multiple datasources. This page will be removed soon.
      </AlertDescription>
    </Alert>
  );
}