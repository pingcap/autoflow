import { getChatMessageSubgraph } from '@/api/chats';
import { useAuth } from '@/components/auth/AuthProvider';
import { type ChatMessageGroup, useChatInfo, useChatMessageStreamState, useCurrentChatController } from '@/components/chat/chat-hooks';
import type { OngoingState } from '@/components/chat/chat-message-controller';
import { AppChatStreamState, type StackVMState } from '@/components/chat/chat-stream-state';
import { NetworkViewer } from '@/components/graph/components/NetworkViewer';
import { useNetwork } from '@/components/graph/useNetwork';
import { PencilIcon } from 'lucide-react';
import Link from 'next/link';
import { useEffect } from 'react';
import useSWR from 'swr';

export function KnowledgeGraphDebugInfo ({ group }: { group: ChatMessageGroup }) {
  const { engine_options } = useChatInfo(useCurrentChatController()) ?? {};
  const auth = useAuth();
  const ongoing = useChatMessageStreamState(group.assistant);
  const kbLinked = engine_options?.knowledge_base?.linked_knowledge_bases;
  const canEdit = !!auth.me?.is_superuser && kbLinked;

  const shouldFetch = (!ongoing || ongoing.finished || couldFetchKnowledgeGraphDebugInfo(ongoing));
  const { data: span, isLoading, mutate, error } = useSWR(
    shouldFetch && `api.chats.get-message-subgraph?id=${group.user.id}`,
    () => getChatMessageSubgraph(group.user.id),
    {
      revalidateOnReconnect: false,
      revalidateOnFocus: false,
      revalidateOnMount: false,
    },
  );

  useEffect(() => {
    if (shouldFetch && !error && !isLoading && !span) {
      mutate(undefined, true);
    }
  }, [span, isLoading, error, shouldFetch]);

  const network = useNetwork(span);

  return (
    <NetworkViewer
      className="my-2 border rounded w-full aspect-video"
      loading={!shouldFetch || isLoading}
      loadingTitle={shouldFetch ? 'Loading knowledge graph...' : 'Waiting knowledge graph request...'}
      network={network}
      Details={
        ({ target, network }) => {
          if (!canEdit) return null;

          if (!kbLinked) return null;

          if (kbLinked.length === 1) {
            return (
              <Link href={`/knowledge-bases/${kbLinked}/knowledge-graph-explorer?query=${encodeURIComponent(`message-subgraph:${group.user.id}`)}`} className="absolute top-2 right-2 text-xs underline">
                <PencilIcon className="w-3 h-3 mr-1 inline-block" />
                Edit graph
              </Link>
            );
          }

          const placeholder = <span className="text-muted-foreground absolute top-2 right-2 text-xs underline cursor-not-allowed">
            <PencilIcon className="w-3 h-3 mr-1 inline-block" />
            Edit graph
          </span>;

          if (!target) return placeholder;

          if (target.type === 'node') {
            const node = network.node(target.id);
            if (!node?.knowledge_base_id) return placeholder;
            return (
              <Link href={`/knowledge-bases/${node.knowledge_base_id}/knowledge-graph-explorer?query=${encodeURIComponent(`message-subgraph:${group.user.id}`)}`} className="absolute top-2 right-2 text-xs underline">
                <PencilIcon className="w-3 h-3 mr-1 inline-block" />
                Edit graph
              </Link>
            );
          } else if (target.type === 'link') {
            const link = network.node(target.id);
            if (!link?.knowledge_base_id) return placeholder;
            return (
              <Link href={`/knowledge-bases/${link.knowledge_base_id}/knowledge-graph-explorer?query=${encodeURIComponent(`message-subgraph:${group.user.id}`)}`} className="absolute top-2 right-2 text-xs underline">
                <PencilIcon className="w-3 h-3 mr-1 inline-block" />
                Edit graph
              </Link>
            );
          }

          return placeholder;
        }
      }
    />
  );
}

function couldFetchKnowledgeGraphDebugInfo (state: OngoingState<AppChatStreamState | StackVMState>) {
  switch (state.state) {
    case AppChatStreamState.GENERATE_ANSWER:
    case AppChatStreamState.FINISHED:
    case AppChatStreamState.RERANKING:
    case AppChatStreamState.SOURCE_NODES:
      return true;
    default:
      return false;
  }
}
