'use client';

import type { Table } from '@tanstack/table-core';
import { createContext, useContext } from 'react';

const DataTableContext = createContext<Table<any> & {
  reload?: () => void
  loading?: boolean
} | null>(null);

export const DataTableProvider = DataTableContext.Provider;

export const DataTableConsumer = DataTableContext.Consumer;

export function useDataTable () {
  const table = useContext(DataTableContext);

  if (!table) {
    throw new Error('useDataTable must be called inside a DataTable');
  }

  return table;
}
