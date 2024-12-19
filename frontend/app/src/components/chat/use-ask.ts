import { useChats } from '@/components/chat/chat-hooks';
import { useGtagFn } from '@/components/gtag-provider';
import { getErrorMessage } from '@/lib/errors';
import { toastError } from '@/lib/ui-error';
import { useRouter } from 'next/navigation';
import { useCallback, useEffect, useRef, useState, useTransition } from 'react';

export function useAsk (onFinish?: () => void) {
  const gtagFn = useGtagFn();
  const { newChat, disabled } = useChats();
  const router = useRouter();
  const [waiting, setWaiting] = useState(false);
  const [transition, startTransition] = useTransition();
  const [engine, setEngine] = useState<string>();
  const engineRef = useRef<string>(undefined);

  const ask = useCallback((message: string, options?: {
    headers?: Record<string, string>;
  }) => {
    setWaiting(true);
    const handleInitialError = (error: unknown) => {
      setWaiting(false);
      toastError('Failed to chat', getErrorMessage(error));
    };

    const controller = newChat(undefined, undefined, { content: message, chat_engine: engineRef.current, headers: options?.headers }, null, gtagFn);

    controller.once('created', chat => {
      controller.off('post-error', handleInitialError);

      setWaiting(false);
      startTransition(() => {
        router.push(`/c/${chat.id}`);
      });
    })
      .once('post-error', handleInitialError);
  }, []);

  useEffect(() => {
    if (!waiting && !transition) {
      onFinish?.();
    }
  }, [waiting, transition]);

  return {
    ask,
    engine,
    disabled,
    setEngine: (engine: string | undefined) => {
      engineRef.current = engine;
      setEngine(engine);
    },
    loading: waiting || transition,
  };
}

export type UseAskReturns = ReturnType<typeof useAsk>;