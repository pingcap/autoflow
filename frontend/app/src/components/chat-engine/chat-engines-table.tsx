'use client';

import { type ChatEngine, createChatEngine, deleteChatEngine, listChatEngines } from '@/api/chat-engines';
import { actions } from '@/components/cells/actions';
import { boolean } from '@/components/cells/boolean';
import { datetime } from '@/components/cells/datetime';
import { mono } from '@/components/cells/mono';
import { DataTableRemote } from '@/components/data-table-remote';
import { useBootstrapStatus } from '@/components/system/BootstrapStatusProvider';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';
import type { ColumnDef } from '@tanstack/react-table';
import { createColumnHelper } from '@tanstack/table-core';
import { AlertTriangleIcon, CopyIcon, TrashIcon } from 'lucide-react';
import Link from 'next/link';
import { toast } from 'sonner';

const helper = createColumnHelper<ChatEngine>();

const columns = [
  helper.accessor('id', { cell: mono }),
  helper.accessor('name', { cell: context => <NameLink chatEngine={context.row.original} /> }),
  helper.accessor('created_at', { cell: datetime }),
  helper.accessor('updated_at', { cell: datetime }),
  helper.accessor('is_default', { cell: boolean }),
  helper.display({
    header: 'Actions',
    cell: actions((chatEngine) => [
      {
        key: 'clone',
        action: async ({ startTransition, router }) => {
          const { name, llm_id, fast_llm_id, engine_options } = chatEngine;
          createChatEngine({
            name: `${name} Copy`, llm_id, fast_llm_id, engine_options,
          })
            .then(newEngine => {
              toast.success('Chat Engine successfully cloned.');
              startTransition(() => {
                router.push(`/chat-engines/${newEngine.id}`);
              });
            });
        },
        icon: <CopyIcon className="size-3" />,
        title: 'Clone',
      },
      {
        key: 'delete',
        action: async ({ table, setDropdownOpen }) => {
          await deleteChatEngine(chatEngine.id);
          table.reload?.();
          setDropdownOpen(false);
        },
        title: 'Delete',
        icon: <TrashIcon className="size-3" />,
        dangerous: {},
      },
    ]),
  }),
] as ColumnDef<ChatEngine>[];

export function ChatEnginesTable () {
  return (
    <DataTableRemote
      columns={columns}
      apiKey="api.chat-engines.list"
      api={listChatEngines}
      idColumn="id"
    />
  );
}

function NameLink ({ chatEngine }: { chatEngine: ChatEngine }) {
  const { need_migration } = useBootstrapStatus();

  const kbNotConfigured = !!need_migration.chat_engines_without_kb_configured?.includes(chatEngine.id);

  return (
    <Link
      className="underline font-mono"
      href={`/chat-engines/${chatEngine.id}`}
    >
      {kbNotConfigured && <TooltipProvider>
        <Tooltip>
          <TooltipTrigger asChild>
            <AlertTriangleIcon className="text-warning inline-flex mr-1 size-3" />
          </TooltipTrigger>
          <TooltipContent className="text-xs" align="start">
            Knowledge Base not configured.
          </TooltipContent>
        </Tooltip>
      </TooltipProvider>}
      {chatEngine.name}
    </Link>
  );
}
