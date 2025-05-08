import { getEmbeddingModel } from '@/api/embedding-models';
import { AdminPageHeading } from '@/components/admin-page-heading';
import { UpdateEmbeddingModelForm } from '@/components/embedding-models/UpdateEmbeddingModelForm';

export default async function Page (props: { params: Promise<{ id: string }> }) {
  const params = await props.params;
  const embeddingModel = await getEmbeddingModel(parseInt(params.id));

  return (
    <>
      <AdminPageHeading
        breadcrumbs={[
          { title: 'Models' },
          { title: 'Embedding Models', url: '/embedding-models', docsUrl: 'https://autoflow.tidb.ai/embedding-model' },
          { title: embeddingModel.name },
        ]}
      />
      <UpdateEmbeddingModelForm embeddingModel={embeddingModel} />
    </>
  );
}
