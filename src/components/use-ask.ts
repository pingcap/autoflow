import { __setMessage } from '@/app/(main)/(public)/c/[id]/internal';
import { handleErrors } from '@/lib/fetch';
import { useSession } from 'next-auth/react';
import { useRouter } from 'next/navigation';
import { useCallback, useState, useTransition } from 'react';
import { mutate } from 'swr'

export function useAsk(onFinish?: () => void) {
  const session = useSession();
  const router = useRouter();
  const [loading, setLoading] = useState(false);
  const [transitioning, startTransition] = useTransition();

  const ask = useCallback((message: string, options?: {
    engine?: number;
    headers?: Record<string, string>;
  }) => {
    startTransition(() => {
      setLoading(true);
      fetch('/api/v1/chats', {
        method: 'post',
        body: JSON.stringify({
          messages: [],
          name: message,
          engine: options?.engine,
        }),
        headers: {...options?.headers},
      }).then(handleErrors)
        .then(res => res.json())
        .then(res => {
          __setMessage(message);
          startTransition(() => {
            onFinish?.()
            router.push(`/c/${encodeURIComponent(res.url_key)}`);
          });
        })
        .finally(() => {
          setLoading(false);
          void mutate(['get', '/api/v1/chats']);
        });
    });
  }, []);

  const disabled = loading || transitioning;

  return {
    ask,
    loading: disabled,
  };
}

export type UseAskReturns = ReturnType<typeof useAsk>;