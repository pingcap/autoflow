'use client';

import { type ApiKey, type CreateApiKeyResponse, deleteApiKey, listApiKeys } from '@/api/api-keys';
import { AdminPageHeading } from '@/components/admin-page-heading';
import { CreateApiKeyForm } from '@/components/api-keys/CreateApiKeyForm';
import { datetime } from '@/components/cells/datetime';
import { CopyButton } from '@/components/copy-button';
import { DangerousActionButton } from '@/components/dangerous-action-button';
import { DataTableRemote } from '@/components/data-table-remote';
import { ManagedDialog } from '@/components/managed-dialog';
import { ManagedDialogClose } from '@/components/managed-dialog-close';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Button } from '@/components/ui/button';
import { DialogContent, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { DataTableConsumer, useDataTable } from '@/components/use-data-table';
import type { CellContext, ColumnDef } from '@tanstack/react-table';
import { createColumnHelper } from '@tanstack/table-core';
import { CircleCheckIcon, PlusIcon, TrashIcon } from 'lucide-react';
import { useState } from 'react';

const helper = createColumnHelper<ApiKey>();

const mono = (cell: CellContext<any, any>) => <span className="font-mono">{cell.getValue()}</span>;

const columns = [
  helper.accessor('api_key_display', { header: 'API Key', cell: mono }),
  helper.accessor('description', { header: 'Description' }),
  helper.accessor('created_at', { header: 'Created At', cell: datetime }),
  helper.accessor('updated_at', { header: 'Updated At', cell: datetime }),
  helper.display({
    header: 'Actions',
    cell: ({ row }) => (
      <span className="flex gap-2 items-center">
        <DeleteButton apiKey={row.original} />
      </span>
    ),
  }),
] as ColumnDef<ApiKey>[];

export default function ChatEnginesPage () {
  const [recentlyCreated, setRecentlyCreated] = useState<CreateApiKeyResponse>();
  return (
    <>
      <AdminPageHeading
        breadcrumbs={[
          { title: 'API Keys' },
        ]}
      />
      {recentlyCreated && (
        <Alert className="max-w-screen-sm" variant="success">
          <CircleCheckIcon />
          <AlertTitle>API Key created</AlertTitle>
          <AlertDescription>
            Please note that your API key will only be shown once. Make sure to save it in a secure location as it won&#39;t be displayed again. Not storing your key safely may result in you needing to generate a new API key.
          </AlertDescription>
          <div className="my-2">
            <p className="px-1 py-0.5 rounded bg-accent text-xs flex items-center">
              <CopyButton text={recentlyCreated.api_key} autoCopy />
              <code className="text-accent-foreground">{recentlyCreated.api_key}</code>
            </p>
          </div>
        </Alert>
      )}
      <DataTableRemote
        before={(
          <ManagedDialog>
            <DialogTrigger asChild>
              <Button className="ml-auto flex">
                Create
                <PlusIcon className="size-4 ml-1" />
              </Button>
            </DialogTrigger>
            <DialogContent>
              <DialogHeader>
                <DialogTitle>Create API Key</DialogTitle>
              </DialogHeader>
              <DataTableConsumer>
                {(table) => (
                  <ManagedDialogClose>
                    {close => (
                      <CreateApiKeyForm
                        onCreated={data => {
                          close();
                          setRecentlyCreated(data);
                          table?.reload?.();
                        }}
                      />
                    )}
                  </ManagedDialogClose>
                )}
              </DataTableConsumer>
            </DialogContent>
          </ManagedDialog>
        )}
        columns={columns}
        apiKey="api.api-keys.list"
        api={listApiKeys}
        idColumn="id"
      />
    </>
  );
}

function DeleteButton ({ apiKey }: { apiKey: ApiKey }) {
  const { reload } = useDataTable();

  return (
    <DangerousActionButton
      action={async () => {
        await deleteApiKey(apiKey.id);
        reload?.();
      }}
      variant="ghost"
      className="text-xs text-destructive hover:text-destructive hover:bg-destructive/20"
    >
      <TrashIcon className="w-3 mr-1" />
      Delete
    </DangerousActionButton>
  );
}
