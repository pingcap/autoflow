'use client';

import type { KnowledgeBase } from '@/api/knowledge-base';
import { createContext, type ReactNode, useContext } from 'react';

const KBContext = createContext<KnowledgeBase>(null as any);

export function KBProvider ({ children, value }: { children: ReactNode, value: KnowledgeBase }) {
  return (
    <KBContext.Provider value={value}>
      {children}
    </KBContext.Provider>
  );
}

export function useKB () {
  return useContext(KBContext);
}
