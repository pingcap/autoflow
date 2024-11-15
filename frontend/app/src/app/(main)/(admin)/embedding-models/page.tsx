'use client';

import { AdminPageHeading } from '@/components/admin-page-heading';
import { ConfigViewer } from '@/components/config-viewer';
import { DateFormat } from '@/components/date-format';
import { EmbeddingModelsTable } from '@/components/embedding-models/EmbeddingModelsTable';
import { OptionDetail } from '@/components/option-detail';

export default function EmbeddingModelPage () {

  return (
    <>
      <AdminPageHeading title="Embedding Models" />
      <EmbeddingModelsTable />
    </>
  );
}
