import { Loader } from '@/components/loader';
import { RowCheckbox } from '@/components/row-checkbox';
import { Button } from '@/components/ui/button';
import { Pagination, PaginationContent, PaginationEllipsis, PaginationItem } from '@/components/ui/pagination';
import { Select, SelectContent, SelectItem, SelectTrigger } from '@/components/ui/select';

import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { TooltipProvider } from '@/components/ui/tooltip';
import { DataTableProvider } from '@/components/use-data-table';
import { getErrorMessage } from '@/lib/errors';
import type { Page, PageParams } from '@/lib/request';
import { cn } from '@/lib/utils';
import { ColumnDef, type ColumnFilter, flexRender, getCoreRowModel, getSortedRowModel, SortingState, Table as ReactTable, useReactTable } from '@tanstack/react-table';
import type { CellContext, PaginationState, RowData } from '@tanstack/table-core';
import { ChevronLeft, ChevronRight } from 'lucide-react';
import { Fragment, type ReactNode, useEffect, useMemo, useState } from 'react';
import useSWR from 'swr';

declare module '@tanstack/table-core' {
  interface ColumnMeta<TData extends RowData, TValue> {
    colSpan?: number | ((context: CellContext<TData, TValue>) => number);
  }
}

export interface PageApiOptions {
  globalFilter: string;
}

interface DataTableRemoteProps<TData, TValue> {
  idColumn: keyof TData;
  apiKey: string;
  api: (page: PageParams, options: PageApiOptions) => Promise<Page<TData>>;
  apiDeps?: unknown[];
  columns: ColumnDef<TData, TValue>[];
  selectable?: boolean;
  batchOperations?: (rows: string[], revalidate: () => void) => ReactNode;
  refreshInterval?: number | ((data: Page<TData> | undefined) => number);
  /**
   * @deprecated
   */
  before?: ReactNode;
  /**
   * @deprecated
   */
  after?: ReactNode;
  toolbar?: (table: ReactTable<TData>) => ReactNode;
  defaultSorting?: SortingState;
}

export function DataTableRemote<TData, TValue> ({
  idColumn,
  api,
  apiKey,
  columns,
  apiDeps = [],
  selectable = false,
  batchOperations,
  refreshInterval,
  before,
  after,
  toolbar,
  defaultSorting = [],
}: DataTableRemoteProps<TData, TValue>) {
  const [pagination, setPagination] = useState<PaginationState>(() => {
    return { pageIndex: 0, pageSize: 10 };
  });
  const [rowSelection, setRowSelection] = useState({});
  const [columnFilters, setColumnFilters] = useState<ColumnFilter[]>([]);
  const [globalFilter, setGlobalFilter] = useState('');
  const [sorting, setSorting] = useState<SortingState>(defaultSorting);

  const idSelection = useMemo(() => {
    return Object.keys(rowSelection);
  }, [rowSelection]);

  // Fetch data.
  const { data, mutate, error, isLoading, isValidating } = useSWR(`${apiKey}?page=${pagination.pageIndex}&size=${pagination.pageSize}${globalFilter && `&query=${globalFilter}`}`, () => api({ page: pagination.pageIndex + 1, size: pagination.pageSize }, { globalFilter }), {
    refreshInterval,
    revalidateOnReconnect: false,
    revalidateOnFocus: false,
    focusThrottleInterval: 1000,
    keepPreviousData: true,
    onError: console.error,
  });

  useEffect(() => {
    void mutate();
  }, [pagination.pageSize, pagination.pageIndex, globalFilter, ...apiDeps]);

  // Column definitions.
  columns = useMemo(() => {
    if (!selectable) {
      return columns;
    }

    return [
      {
        id: 'select',
        header: ({ table }) => (
          <RowCheckbox
            onClick={table.getToggleAllRowsSelectedHandler()}
            checked={table.getIsAllRowsSelected()}
            indeterminate={table.getIsSomeRowsSelected()}
          />
        ),
        cell: ({ row }) => (
          <div>
            <RowCheckbox
              onClick={row.getToggleSelectedHandler()}
              checked={row.getIsSelected()}
              indeterminate={row.getIsSomeSelected()}
              disabled={!row.getCanSelect()}
            />
          </div>
        ),
      },
      ...columns,
    ];
  }, [columns, selectable]);

  const table = useReactTable({
    data: data?.items ?? [],
    columns,
    state: {
      sorting,
      pagination,
      rowSelection,
      columnFilters,
      globalFilter,
    },
    pageCount: data ? data.pages : 1,
    manualPagination: true,
    manualFiltering: true,
    enableRowSelection: selectable,
    enableMultiRowSelection: selectable,
    enableColumnFilters: true,
    enableGlobalFilter: true,
    onSortingChange: async (val) => {
      await mutate();
      setSorting(val);
    },
    onPaginationChange: setPagination,
    onRowSelectionChange: setRowSelection,
    onColumnFiltersChange: setColumnFilters,
    onGlobalFilterChange: setGlobalFilter,
    getSortedRowModel: getSortedRowModel(),
    getCoreRowModel: getCoreRowModel(),
    getRowId: item => String(item[idColumn]),
  });

  return (
    <DataTableProvider
      value={{
        ...table,
        reload: () => { mutate(); },
        loading: isLoading,
      }}
    >
      {before}
      {toolbar ? toolbar(table) : null}
      <TooltipProvider>
        <div className="rounded-md border relative">
          <Table className="text-xs whitespace-nowrap">
            <TableHeader>
              {table.getHeaderGroups().map((headerGroup) => (
                <TableRow key={headerGroup.id}>
                  {headerGroup.headers.map((header) => {
                    return (
                      <TableHead key={header.id} colSpan={header.colSpan}>
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
            </TableHeader>
            <TableBody>
              {table.getRowModel().rows?.length ? (
                table.getRowModel().rows.map((row) => (
                  <TableRow
                    key={row.id}
                    data-state={row.getIsSelected() && 'selected'}
                  >
                    {row.getVisibleCells().map((cell) => {
                      // Col span for advanced customization.
                      const span = getColSpan(cell.column.columnDef, cell.getContext());

                      if (span === 0) {
                        return <Fragment key={cell.id} />;
                      }

                      return (
                        <TableCell key={cell.id} colSpan={span}>
                          {flexRender(cell.column.columnDef.cell, cell.getContext())}
                        </TableCell>
                      );
                    })}
                  </TableRow>
                ))
              ) : (
                <TableRow>
                  <TableCell colSpan={columns.length} className={cn('h-24 text-center', !!error && 'text-destructive')}>
                    {error
                      ? `Failed to load data: ${getErrorMessage(error)}`
                      : 'Empty List'}
                  </TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>
          <Loader loading={isLoading || isValidating} />
        </div>
        <div className="flex w-full gap-2 py-4">
          {selectable && (
            <>
              <span className="text-xs text-secondary-foreground">
                Selected {Object.keys(rowSelection).length} rows
              </span>
              {batchOperations?.(idSelection, () => mutate())}
            </>
          )}
          <TablePagination className="mx-0 ml-auto w-max" loading={isLoading} table={table} />
        </div>
      </TooltipProvider>
      {after}
    </DataTableProvider>
  );
}

function getSortingSearchString (sorting: SortingState) {
  return sorting.map(({ id, desc }) => `${id}:${desc ? 'desc' : 'asc'}`).join(',');
}

const sizes = [10, 20, 50, 100];

function TablePagination ({ className, limit = 4, loading, table }: { className?: string, limit?: number, loading: boolean, table: ReactTable<any> }) {
  const options = table.getPageOptions();
  const pagination = table.getState().pagination;

  const min = Math.max(pagination.pageIndex - limit / 2, 0);
  const max = Math.min(min + limit + 1, options.length - 1);

  if (min >= max) {
    return <span className={className} />;
  }

  return (
    <Pagination className={className}>
      <Select value={String(pagination.pageSize)} onValueChange={value => table.setPageSize(Number(value))}>
        <SelectTrigger className="w-max">
          {pagination.pageSize} / page
        </SelectTrigger>
        <SelectContent>
          {sizes.map(size => (
            <SelectItem key={size} value={String(size)}>
              {size}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
      <PaginationContent>
        <PaginationItem>
          <Button variant="ghost" size="icon" disabled={loading || !table.getCanPreviousPage()} onClick={() => table.previousPage()}>
            <ChevronLeft />
          </Button>
        </PaginationItem>
        {min > 0 && (
          <PaginationItem>
            <Button variant="ghost" size="icon" disabled={loading} onClick={() => table.setPageIndex(0)}>
              1
            </Button>
          </PaginationItem>
        )}
        {min > 1 && (
          <PaginationItem>
            <PaginationEllipsis />
          </PaginationItem>
        )}
        {steps(min, max).map((page) => (
          <PaginationItem key={page}>
            <Button
              variant={page === pagination.pageIndex ? 'outline' : 'ghost'}
              disabled={loading}
              size="icon"
              onClick={() => table.setPageIndex(page)}
            >
              {page + 1}
            </Button>
          </PaginationItem>
        ))}
        {(max < options.length - 2) && (
          <PaginationItem>
            <PaginationEllipsis />
          </PaginationItem>
        )}
        {(max < options.length - 1) && (
          <PaginationItem>
            <Button variant="ghost" size="icon" disabled={loading} onClick={() => table.setPageIndex(options.length - 1)}>
              {options.length}
            </Button>
          </PaginationItem>
        )}
        <PaginationItem>
          <Button variant="ghost" size="icon" disabled={loading || !table.getCanNextPage()} onClick={() => table.nextPage()}>
            <ChevronRight />
          </Button>
        </PaginationItem>
      </PaginationContent>
    </Pagination>
  );
}

function steps (from: number, to: number) {
  if (from >= to) {
    return [];
  }
  let arr = new Array(to - from + 1);
  for (let i = from; i <= to; i++) {
    arr[i - from] = i;
  }

  return arr;
}

function getColSpan<TData extends RowData, TValue> (columnDef: ColumnDef<TData, TValue>, context: CellContext<TData, TValue>) {
  const colSpan = columnDef.meta?.colSpan;
  if (colSpan == null) {
    return undefined;
  }
  if (typeof colSpan === 'number') {
    return colSpan;
  }
  return colSpan(context);
}
