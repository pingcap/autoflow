import { deleteKnowledgeBase, getKnowledgeBaseLinkedChatEngines, type KnowledgeBaseSummary } from '@/api/knowledge-base';
import { DangerousActionButton } from '@/components/dangerous-action-button';
import { mutateKnowledgeBases } from '@/components/knowledge-base/hooks';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardFooter, CardHeader } from '@/components/ui/card';
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuSeparator, DropdownMenuTrigger } from '@/components/ui/dropdown-menu';
import { Separator } from '@/components/ui/separator';
import { Skeleton } from '@/components/ui/skeleton';
import { cn } from '@/lib/utils';
import { AlertTriangleIcon, Book, Ellipsis, TriangleAlertIcon } from 'lucide-react';
import { useRouter } from 'next/navigation';
import { ReactNode, startTransition, useState } from 'react';
import useSWR from 'swr';

export function KnowledgeBaseCard ({ knowledgeBase, children }: { knowledgeBase: KnowledgeBaseSummary, children?: ReactNode }) {
  const router = useRouter();
  const [dropdownOpen, setDropdownOpen] = useState(false);
  const { data: linkedChatEngines } = useSWR(`api.knowledge-bases.${knowledgeBase.id}.linked-chat-engines`, () => getKnowledgeBaseLinkedChatEngines(knowledgeBase.id));

  const handleCardClick = () => {
    startTransition(() => {
      router.push(`/knowledge-bases/${knowledgeBase.id}`);
    });
  };

  const handleMenuItemSettingSelect = (event: Event) => {
    event.preventDefault();
    startTransition(() => {
      router.push(`/knowledge-bases/${knowledgeBase.id}/settings`);
    });
  };

  const handleDelete = async () => {
    await deleteKnowledgeBase(knowledgeBase.id);
    await mutateKnowledgeBases();
    setDropdownOpen(false);
  };

  return (
    <Card className={cn('cursor-pointer transition-colors hover:bg-muted/50 max-h-64', dropdownOpen && 'bg-muted/50')} onClick={handleCardClick}>
      <CardHeader className="p-4">
        <div className="flex justify-start space-x-4">
          <div className="flex border w-10 h-10 rounded-md justify-center items-center bg-secondary">
            <Book className="size-5" />
          </div>
          <div className="flex-1 space-y-1">
            <h4 className="text-sm font-semibold">{knowledgeBase.name}</h4>
            <div className="flex items-center text-xs text-muted-foreground">
              <span>{knowledgeBase.documents_total ?? 0} documents</span>
              <span className="shrink-0 mx-0.5 px-1">·</span>
              <span>{(knowledgeBase.data_sources_total ?? 0) || <><AlertTriangleIcon className="size-3 inline-flex" /> No</>} data sources</span>
            </div>
            <div className="flex items-center text-xs text-muted-foreground">
              <span>{linkedChatEngines?.length ?? <Skeleton className="inline-flex h-3 w-6 rounded" />} linked chat engines</span>
            </div>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <CardDescription className="text-xs line-clamp-2 text-muted-foreground">
          {knowledgeBase.description}
        </CardDescription>
      </CardContent>
      <CardFooter className="flex justify-between items-center text-sm p-2">
        <div className="flex items-center gap-2 pl-2">
          {knowledgeBase.index_methods.map(m => <Badge key={m} variant="secondary">{m}</Badge>)}
        </div>
        <div>
          <Separator orientation="vertical" />
          <DropdownMenu open={dropdownOpen} onOpenChange={setDropdownOpen}>
            <DropdownMenuTrigger asChild>
              <Button variant="ghost" size="sm" onClick={event => event.stopPropagation()}>
                <Ellipsis className="size-5" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent className="w-56" align="end" alignOffset={-9} onClick={event => event.stopPropagation()}>
              <DropdownMenuItem onSelect={handleMenuItemSettingSelect}>Settings</DropdownMenuItem>
              <DropdownMenuSeparator />
              <DangerousActionButton
                action={handleDelete}
                asChild
                actionDisabled={(linkedChatEngines?.length ?? 0) > 0}
                actionDisabledReason={<Alert variant="warning">
                  <TriangleAlertIcon />
                  <AlertTitle>Cannot delete this Knowledge Base</AlertTitle>
                  <AlertDescription>This Knowledge Base was linked to at least one Chat Engine(s). Please unlink all Chat Engines to continue.</AlertDescription>
                </Alert>}
              >
                <DropdownMenuItem
                  className="text-destructive focus:text-destructive focus:bg-destructive/10"
                  disabled={linkedChatEngines == null}
                  onSelect={event => event.preventDefault()}
                >
                  Delete
                </DropdownMenuItem>
              </DangerousActionButton>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </CardFooter>
    </Card>
  );
}

export function KnowledgeBaseCardPlaceholder () {
  return (
    <Card className="max-h-64">
      <CardHeader className="p-4">
        <div className="flex justify-start space-x-4">
          <Skeleton className="size-10" />
          <div className="flex-1 space-y-1">
            <h4 className="text-sm font-semibold"><Skeleton className="w-28 h-[1em] mt-[0.25em] mb-[0.5em]" /></h4>
            <div className="flex items-center text-xs text-muted-foreground gap-2">
              <Skeleton className="w-16 h-[1em]" />
              <Skeleton className="w-24 h-[1em]" />
            </div>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <div className="text-muted-foreground text-xs line-clamp-2">
          <Skeleton className="w-full h-[1em] my-[0.25em]" />
          <Skeleton className="w-[70%] h-[1em] my-[0.25em]" />
        </div>
      </CardContent>
      <CardFooter className="flex items-center text-sm p-2">
        <div className="flex items-center gap-2 pl-2">
          <Skeleton className="rounded-full w-16 h-[1.25em]" />
          <Skeleton className="rounded-full w-24 h-[1.25em]" />
        </div>
      </CardFooter>
    </Card>
  );
}