'use client';

import { type Datasource, deleteDatasource } from '@/api/datasources';
import { DangerousActionButton } from '@/components/dangerous-action-button';
import { UpdateDatasourceForm } from '@/components/datasource/update-datasource-form';
import { ManagedDialog } from '@/components/managed-dialog';
import { ManagedPanelContext } from '@/components/managed-panel';
import { Button } from '@/components/ui/button';
import { Card, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover';
import { FileDownIcon, GlobeIcon, PaperclipIcon } from 'lucide-react';
import { useRouter } from 'next/navigation';

export function DatasourceCard ({ knowledgeBaseId, datasource }: { knowledgeBaseId: number, datasource: Datasource }) {
  const router = useRouter();

  return (
    <Card key={datasource.id}>
      <CardHeader className="p-3">
        <CardTitle className="text-base">{datasource.name}</CardTitle>
        <CardDescription className="text-xs">
          <DatasourceCardDetails datasource={datasource} />
        </CardDescription>
      </CardHeader>
      <CardFooter className="gap-2 p-3 pt-0">
        <ManagedDialog>
          <DialogTrigger asChild>
            <Button variant="ghost" size="sm">Configure</Button>
          </DialogTrigger>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>Configure Datasource</DialogTitle>
              <DialogDescription />
            </DialogHeader>
            <ManagedPanelContext.Consumer>
              {({ setOpen }) => (
                <UpdateDatasourceForm
                  knowledgeBaseId={knowledgeBaseId}
                  datasource={datasource}
                  onUpdated={() => {
                    router.refresh();
                    setOpen(false);
                  }}
                />
              )}
            </ManagedPanelContext.Consumer>
          </DialogContent>
        </ManagedDialog>
        <DangerousActionButton
          action={async () => {
            await deleteDatasource(knowledgeBaseId, datasource.id);
          }}
          asChild
          dialogTitle={`Confirm to delete the datasource ${datasource.name} #${datasource.id}`}
          dialogDescription={<>All <b>documents</b>, <b>chunks</b>, <b>entities</b> and <b>relationships</b> related to this datasource will be <b>deleted</b>. This action cannot be undone.</>}
        >
          <Button variant="ghost" className="hover:text-destructive hover:bg-destructive/10" size="sm">Delete</Button>
        </DangerousActionButton>
      </CardFooter>
    </Card>
  );
}

function DatasourceCardDetails ({ datasource }: { datasource: Datasource }) {
  return (
    <span className="flex gap-1 items-center">
      {(() => {
        switch (datasource.data_source_type) {
          case 'web_sitemap':
            return <GlobeIcon className="size-3" />;
          case 'web_single_page':
            return <FileDownIcon className="size-3" />;
          case 'file':
            return <PaperclipIcon className="size-3" />;
        }
      })()}
      <span>
        {(() => {
          switch (datasource.data_source_type) {
            case 'web_sitemap':
              return datasource.config.url;
            case 'web_single_page':
              return datasource.config.urls.join(', ');
            case 'file':
              if (datasource.config.length === 1) {
                return datasource.config[0].file_name;
              } else {
                return (
                  <>
                    {datasource.config[0]?.file_name}
                    {(datasource.config.length > 1) && <Popover>
                      <PopoverTrigger className="ml-2 font-medium">
                        +{datasource.config.length - 1} files
                      </PopoverTrigger>
                      <PopoverContent className="flex flex-wrap gap-2 text-xs text-muted-foreground">
                        {datasource.config.slice(1).map(file => (
                          <span key={file.file_id}>{file.file_name}</span>
                        ))}
                      </PopoverContent>
                    </Popover>}
                  </>
                );
              }
          }
        })()}
      </span>
    </span>
  );
}
