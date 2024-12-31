import type { ChatMessageSource } from '@/api/chats';
import type { OngoingState } from '@/components/chat/chat-message-controller';

export type { ChatEngineOptions } from '@/api/chat-engines';

const truncateUrl = (url: string, maxLength: number = 20): string => {
  if (!url || url.length <= maxLength) return url;
  const start = url.substring(0, maxLength / 2);
  const end = url.substring(url.length - maxLength / 2);
  return `${start}...${end}`;
};

export function parseSource (uri?: string) {
  if (!uri) {
    return 'Unknown';
  }
  if (/^https:\/\//.test(uri)) {
    return new URL(uri).hostname;
  } else {
    return truncateUrl(uri);
  }
}

export function parseHref (source: ChatMessageSource): { href: string, download?: string, target?: HTMLAnchorElement['target'] } {
  if (/^https?:\/\//.test(source.source_uri)) {
    return { href: source.source_uri, target: '_blank' };
  } else if (source.source_uri.startsWith('uploads/')) {
    return { href: `/api/v1/documents/${source.id}/download`, download: source.source_uri.slice(source.source_uri.lastIndexOf('/') + 1) };
  } else {
    return { href: 'javascript:void(0)' };
  }
}

export function isNotFinished (ongoing: OngoingState<any> | undefined) {
  return !!ongoing && !ongoing.finished;
}
