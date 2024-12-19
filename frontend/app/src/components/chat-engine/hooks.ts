import { listChatEngines } from '@/api/chat-engines';
import { listAllHelper } from '@/lib/request';
import useSWR from 'swr';

export function useAllChatEngines () {
  return useSWR('api.chat-engines.list-all', () => listAllHelper(listChatEngines, 'id'));
}