'use client';

import { type Feedback, FeedbackType, listFeedbacks } from '@/api/feedbacks';
import { datetime } from '@/components/cells/datetime';
import { mono } from '@/components/cells/mono';
import { DataTableRemote } from '@/components/data-table-remote';
import type { ColumnDef } from '@tanstack/react-table';
import { createColumnHelper } from '@tanstack/table-core';
import { ThumbsDownIcon, ThumbsUpIcon } from 'lucide-react';
import Link from 'next/link';

const helper = createColumnHelper<Feedback>();

const columns = [
  helper.accessor('id', { header: 'ID', cell: mono }),
  helper.accessor('feedback_type', {
    header: 'TYPE',
    cell: (cell) => {
      const type = cell.getValue();
      switch (type) {
        case FeedbackType.like:
          return (<span className="flex gap-2 items-center text-success"><ThumbsUpIcon className="size-4" /> LIKE</span>);
        case FeedbackType.dislike:
          return (<span className="flex gap-2 items-center text-destructive"><ThumbsDownIcon className="size-4" /> DISLIKE</span>);
      }
    },
  }),
  helper.accessor('origin', { header: 'FEEDBACK ORIGIN', cell: mono }),
  helper.accessor('chat_origin', { header: 'CHAT ORIGIN', cell: mono }),
  helper.display({
    id: 'chat',
    header: 'QUESTION',
    cell: ({ row }) =>
      <Link className="underline" href={`/c/${row.original.chat_id}#${row.original.chat_message_id}`}>
        <b>{row.original.chat_title}</b> <span className="text-muted-foreground">{row.original.chat_id}#{row.original.chat_message_id}</span>
      </Link>,
  }),
  helper.accessor('chat_message_content', {
    header: 'CONTENT',
    cell: cell => <>{cell.getValue().slice(0, 50)}... <span className="text-muted-foreground">({cell.getValue().length + ' characters'})</span></>,
  }),
  helper.accessor('comment', { header: 'COMMENT', cell: mono }),
  helper.accessor('user_email', { header: 'USER', cell: mono }),
  helper.accessor('created_at', { header: 'CREATED AT', cell: datetime }),
] as ColumnDef<Feedback>[];

export function FeedbacksTable () {
  return (
    <DataTableRemote
      columns={columns}
      apiKey="api.feedbacks.list"
      api={listFeedbacks}
      idColumn="id"
    />
  );
}
