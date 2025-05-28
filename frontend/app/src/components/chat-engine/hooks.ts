import { listChatEngines, listPublicChatEngines } from '@/api/chat-engines';
import { listAllHelper } from '@/lib/request';
import useSWR from 'swr';

export function useAllChatEngines (onlyPublic: boolean = false) {
  return useSWR(onlyPublic ? 'api.chat-engines.list-all-public' : 'api.chat-engines.list-all', () => listAllHelper(onlyPublic ? listPublicChatEngines : listChatEngines, 'id'));
}