'use client';

import type { BootstrapStatus } from '@/api/system';
import { createContext, type ReactNode, useContext } from 'react';

const BootstrapStatusContext = createContext<BootstrapStatus>({
  required: {
    knowledge_base: false,
    default_embedding_model: false,
    default_chat_engine: false,
    default_llm: false,
  },
  optional: {
    langfuse: false,
    default_reranker: false,
  },
  need_migration: {
    chat_engines_without_kb_configured: [-1],
  },
});

export function BootstrapStatusProvider ({ bootstrapStatus, children }: { bootstrapStatus: BootstrapStatus, children: ReactNode }) {
  return <BootstrapStatusContext.Provider value={bootstrapStatus}>{children}</BootstrapStatusContext.Provider>;
}

export function useBootstrapStatus () {
  return useContext(BootstrapStatusContext);
}