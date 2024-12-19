'use client';

import { type FeedbackParams, postFeedback } from '@/api/chats';
import { useState } from 'react';

export interface UseMessageFeedbackReturns {
  feedbackData: FeedbackParams | undefined;
  disabled: boolean;

  feedback (action: 'like' | 'dislike', comment: string): Promise<void>;
}

export function useMessageFeedback (messageId: number | undefined, enabled: boolean = true): UseMessageFeedbackReturns {
  const [feedback, setFeedback] = useState<FeedbackParams>();
  const isLoading = false;
  const isValidating = false;
  const [acting, setActing] = useState(false);
  const disabled = messageId == null && isValidating || isLoading || acting || !enabled;

  return {
    feedbackData: feedback,
    disabled,
    feedback: async (action, /* detail, */ comment) => {
      if (!messageId) {
        return;
      }
      setActing(true);
      await postFeedback(messageId, { feedback_type: action, comment }).finally(() => setActing(false));
      setFeedback({ feedback_type: action, comment });
    },
  };
}
