'use client';

import { Loader } from '@/components/loader';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { DataTableProvider } from '@/components/use-data-table';
import { cn } from '@/lib/utils';
import { ColumnDef, flexRender, getCoreRowModel, useReactTable } from '@tanstack/react-table';
import type { ReactNode } from 'react';

interface DataTableClassNames {
  table?: string;
  tr?: string;
  td?: string;
  headTd?: string;
  headTr?: string;
}

interface DataTableProps<TData, TValue> {
  className?: string;
  before?: ReactNode;
  after?: ReactNode;
  hideHeader?: boolean;
  columns: ColumnDef<TData, TValue>[];
  data: TData[];
  classNames?: DataTableClassNames;
  loading?: boolean
}

export function DataTable<TData, TValue> ({
  className,
  hideHeader,
  columns,
  data,
  before,
  after,
  classNames,
  loading,
}: DataTableProps<TData, TValue>) {
  const table = useReactTable({
    data,
    columns,
    getCoreRowModel: getCoreRowModel(),
  });

  return (
    <DataTableProvider value={table}>
      {before}
      <div className={cn('rounded-md border relative', className)}>
        <Table className="text-xs whitespace-nowrap">
          {!hideHeader && <TableHeader>
            {table.getHeaderGroups().map((headerGroup) => (
              <TableRow key={headerGroup.id} className={classNames?.headTd}>
                {headerGroup.headers.map((header) => {
                  return (
                    <TableHead key={header.id} className={classNames?.headTr}>
                      {header.isPlaceholder
                        ? null
                        : flexRender(
                          header.column.columnDef.header,
                          header.getContext(),
                        )}
                    </TableHead>
                  );
                })}
              </TableRow>
            ))}
          </TableHeader>}
          <TableBody>
            {table.getRowModel().rows?.length ? (
              table.getRowModel().rows.map((row) => (
                <TableRow
                  key={row.id}
                  className={classNames?.td}
                  data-state={row.getIsSelected() && 'selected'}
                >
                  {row.getVisibleCells().map((cell) => (
                    <TableCell key={cell.id} className={classNames?.td}>
                      {flexRender(cell.column.columnDef.cell, cell.getContext())}
                    </TableCell>
                  ))}
                </TableRow>
              ))
            ) : (
              <TableRow>
                <TableCell colSpan={columns.length} className="h-24 text-center">
                  No results.
                </TableCell>
              </TableRow>
            )}
          </TableBody>
        </Table>
        <Loader loading={!!loading} />
      </div>
      {after}
    </DataTableProvider>
  );
}
