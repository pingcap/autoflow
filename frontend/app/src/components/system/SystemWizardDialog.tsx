'use client';

import { getAllSiteSettings } from '@/api/site-settings';
import { isBootstrapStatusPassed } from '@/api/system';
import { useAuth } from '@/components/auth/AuthProvider';
import { CreateDatasourceForm } from '@/components/datasource/CreateDatasourceForm';
import { DatasourceTypeTabs } from '@/components/datasource/DatasourceTypeTabs';
import type { DatasourceType } from '@/components/datasource/types';
import { CreateEmbeddingModelForm } from '@/components/embedding-models/CreateEmbeddingModelForm';
import { CreateLLMForm } from '@/components/llm/CreateLLMForm';
import { useRefresh } from '@/components/nextjs/app-router-hooks';
import { NextLink } from '@/components/nextjs/NextLink';
import { CreateRerankerForm } from '@/components/reranker/CreateRerankerForm';
import { LangfuseSettings } from '@/components/settings/IntegrationsSettings';
import { Signin } from '@/components/signin';
import { useBootstrapStatus } from '@/components/system/BootstrapStatusProvider';
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from '@/components/ui/accordion';
import { Button } from '@/components/ui/button';
import { Dialog, DialogDescription, DialogFooter, DialogHeader, DialogOverlay, DialogPortal, DialogTitle } from '@/components/ui/dialog';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Skeleton } from '@/components/ui/skeleton';
import { cn } from '@/lib/utils';
import * as DialogPrimitive from '@radix-ui/react-dialog';
import { ArrowRightIcon, CircleAlertIcon, CircleCheckIcon, Loader2Icon } from 'lucide-react';
import { usePathname } from 'next/navigation';
import * as React from 'react';
import { type ReactElement, useState } from 'react';
import useSWR from 'swr';

export function SystemWizardDialog () {
  const pathname = usePathname();
  const bootstrapStatus = useBootstrapStatus();
  const [open, setOpen] = useState(() => !isBootstrapStatusPassed(bootstrapStatus));
  const [refreshing, refresh] = useRefresh();
  const [datasourceType, setDatasourceType] = useState<DatasourceType>('file');
  const { me, isValidating, reload } = useAuth();

  const configured = isBootstrapStatusPassed(bootstrapStatus);
  const { data: siteSettings, isLoading: isSiteSettingsLoading, isValidating: isSiteSettingsValidating, mutate: mutateSiteSettings } = useSWR(me && 'api.site-settings.get-all', () => getAllSiteSettings(), { revalidateOnFocus: false });

  let el: ReactElement;

  if (!me) {
    el = (
      <>
        <DialogHeader>
          <DialogTitle>Almost there...</DialogTitle>
          <DialogDescription>Your app is not fully configured yet. Please login to complete the setup process.</DialogDescription>
        </DialogHeader>
        {isValidating
          ? <Loader2Icon className="mt-4 text-muted-foreground size-4 animate-spin repeat-infinite" />
          : <div className="mt-4 rounded border p-4">
            <Signin noRedirect onLoggedIn={() => reload()} />
            <p className="text-sm text-muted-foreground mt-2">You can find your initial username and password in the server console output.</p>
          </div>}
      </>)
    ;
  } else {
    el = (
      <>
        <DialogHeader>
          <DialogTitle>
            {configured
              ? 'Successfully configured'
              : 'Almost there...'}
          </DialogTitle>
          <DialogDescription>
            {configured
              ? 'Congratulation! You have finished the setup process, it may take minutes to import datasource and build index.'
              : 'Your app is not fully configured yet. Please complete the setup process.'}
          </DialogDescription>
          <Accordion type="single" collapsible>
            <AccordionItem value="required.default_llm">
              <AccordionTrigger disabled={bootstrapStatus.required.default_llm}>
                  <span>
                    <StatusIcon flag={bootstrapStatus.required.default_llm} />
                    Setup default LLM
                  </span>
              </AccordionTrigger>
              {!bootstrapStatus.required.default_llm && <AccordionContent className="px-4">
                <CreateLLMForm
                  transitioning={refreshing}
                  onCreated={() => {
                    refresh();
                  }}
                />
              </AccordionContent>}
            </AccordionItem>
            <AccordionItem value="required.default_embedding_model">
              <AccordionTrigger disabled={bootstrapStatus.required.default_embedding_model}>
                  <span>
                    <StatusIcon flag={bootstrapStatus.required.default_embedding_model} />
                    Setup default Embedding Model
                  </span>
              </AccordionTrigger>
              {!bootstrapStatus.required.default_embedding_model && <AccordionContent className="px-4">
                <CreateEmbeddingModelForm
                  transitioning={refreshing}
                  onCreated={() => {
                    refresh();
                  }}
                />
              </AccordionContent>}
            </AccordionItem>
            <AccordionItem value="required.datasource">
              <AccordionTrigger disabled={bootstrapStatus.required.datasource}>
                  <span>
                    <StatusIcon flag={bootstrapStatus.required.datasource} />
                    Setup Datasource
                  </span>
              </AccordionTrigger>
              {!bootstrapStatus.required.datasource && (
                <AccordionContent className="px-4">
                  <DatasourceTypeTabs className="mb-4" type={datasourceType} onTypeChange={setDatasourceType} />
                  <CreateDatasourceForm
                    excludesLLM
                    type={datasourceType}
                    transitioning={refreshing}
                    onCreated={() => {
                      refresh();
                    }}
                  />
                </AccordionContent>
              )}
            </AccordionItem>
            <AccordionItem value="optional.langfuse">
              <AccordionTrigger>
                  <span>
                    <StatusIcon flag={bootstrapStatus.optional.langfuse} optional />
                    [Optional] Setup Langfuse
                  </span>
              </AccordionTrigger>
              <AccordionContent className="px-4">
                {isSiteSettingsLoading && (
                  <div className="space-y-2">
                    <Skeleton className="w-[70%] h-4" />
                    <Skeleton className="w-[30%] h-4" />
                    <Skeleton className="w-[50%] h-4" />
                  </div>
                )}
                {siteSettings && <LangfuseSettings schema={siteSettings} hideTitle disabled={isSiteSettingsLoading || isSiteSettingsValidating} onChanged={() => mutateSiteSettings()} />}
              </AccordionContent>
            </AccordionItem>
            <AccordionItem value="optional.default_reranker">
              <AccordionTrigger disabled={bootstrapStatus.optional.default_reranker}>
                  <span>
                    <StatusIcon flag={bootstrapStatus.optional.default_reranker} optional />
                    [Optional] Setup default Reranker
                  </span>
              </AccordionTrigger>
              {!bootstrapStatus.optional.default_reranker && <AccordionContent className="px-4">
                <CreateRerankerForm
                  transitioning={refreshing}
                  onCreated={() => {
                    refresh();
                  }}
                />
              </AccordionContent>}
            </AccordionItem>
          </Accordion>
        </DialogHeader>
        {configured && <DialogFooter className="mt-2">
          <NextLink
            variant="ghost"
            href="/index-progress"
            onClick={() => {
              setOpen(false);
            }}
          >
            View Index Progress
            <ArrowRightIcon className="size-4 ml-1" />
          </NextLink>
          <Button onClick={() => setOpen(false)}>
            OK
          </Button>
        </DialogFooter>}
      </>
    );
  }

  return (
    <Dialog open={open && pathname !== '/auth/login' && !(!me && configured)}>
      <DialogPortal>
        <DialogOverlay />
        <DialogPrimitive.Content
          className={cn(
            'fixed left-[50%] top-[50%] z-50 grid w-full max-w-screen-sm translate-x-[-50%] translate-y-[-50%] gap-4 border bg-background p-6 shadow-lg duration-200 data-[state=open]:animate-in data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0 data-[state=closed]:zoom-out-95 data-[state=open]:zoom-in-95 data-[state=closed]:slide-out-to-left-1/2 data-[state=closed]:slide-out-to-top-[48%] data-[state=open]:slide-in-from-left-1/2 data-[state=open]:slide-in-from-top-[48%] sm:rounded-lg',
          )}
        >
          <ScrollArea className="max-h-[80vh]">
            <div className="p-1">
              {el}
            </div>
          </ScrollArea>
        </DialogPrimitive.Content>
      </DialogPortal>
    </Dialog>
  );
}

function StatusIcon ({ flag, optional }: { flag: boolean, optional?: boolean }) {
  if (flag) {
    return (<CircleCheckIcon className="inline-block size-4 mr-2 text-green-500" />);
  } else if (optional) {
    return (<CircleAlertIcon className="inline-block size-4 mr-2 text-accent-foreground" />);
  } else {
    return (<CircleAlertIcon className="inline-block size-4 mr-2 text-yellow-500" />);
  }
}
