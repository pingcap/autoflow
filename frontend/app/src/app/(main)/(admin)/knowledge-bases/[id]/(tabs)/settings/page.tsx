import { cachedGetKnowledgeBaseById } from '@/app/(main)/(admin)/knowledge-bases/[id]/api';
import { KnowledgeBaseSettingsForm } from '@/components/knowledge-base/knowledge-base-settings-form';

export default async function KnowledgeBaseSettingsPage(props: { params: Promise<{ id: string }> }) {
  const params = await props.params;
  const id = parseInt(decodeURIComponent(params.id));
  const kb = await cachedGetKnowledgeBaseById(id);

  return (
    <KnowledgeBaseSettingsForm knowledgeBase={kb} />
  );
}
