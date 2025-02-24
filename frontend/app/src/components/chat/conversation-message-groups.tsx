import { type ChatMessageGroup, useChatInfo, useChatPostState, useCurrentChatController } from '@/components/chat/chat-hooks';
import { DebugInfo } from '@/components/chat/debug-info';
import { MessageAnnotationHistory } from '@/components/chat/message-annotation-history';
import { StackVMMessageAnnotationHistory } from '@/components/chat/message-annotation-history-stackvm';
import { MessageAnswer } from '@/components/chat/message-answer';
import { MessageAutoScroll } from '@/components/chat/message-auto-scroll';
import { MessageContextSources } from '@/components/chat/message-content-sources';
import { MessageError } from '@/components/chat/message-error';
import { MessageOperations } from '@/components/chat/message-operations';
import { MessageRecommendQuestions } from '@/components/chat/message-recommend-questions';
import { MessageSection } from '@/components/chat/message-section';
import { Button } from '@/components/ui/button';
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible';
import { MessageVerify } from '@/experimental/chat-verify-service/message-verify';
import { cn } from '@/lib/utils';
import { InfoIcon } from 'lucide-react';
import { useEffect, useState } from 'react';
import './conversation-message-groups.scss';

const isWidgetEnv = !!process.env.NEXT_PUBLIC_IS_WIDGET;

export function ConversationMessageGroups ({ groups }: { groups: ChatMessageGroup[] }) {
  const controller = useCurrentChatController();
  const { params, initialized } = useChatPostState(useCurrentChatController());

  useEffect(() => {
    if (!isWidgetEnv) {
      const scroll = () => {
        setTimeout(() => {
          window.scrollTo({
            left: 0,
            top: document.body.scrollHeight,
            behavior: 'smooth',
          });
        }, 100);
      };

      controller
        .on('post', scroll)
        .on('post-initialized', scroll);

      return () => {
        controller
          .off('post', scroll)
          .off('post-initialized', scroll);
      };
    }
  }, [controller]);

  return (
    <div className="space-y-8 pb-16">
      {groups.map((group, index) => (
        <ConversationMessageGroup
          key={group.user.id}
          group={group}
          isLastGroup={index === groups.length - 1}
        />
      ))}
      {!!params && !initialized && (
        <section
          className={cn('opacity-50 pointer-events-none space-y-6 p-4 pt-12 border-b pb-10 last-of-type:border-b-0 last-of-type:border-pb-4')}
        >
          <div className="relative pr-12">
            <h2 className="text-2xl font-normal">{params.content}</h2>
          </div>
        </section>
      )}
    </div>
  );
}

function ConversationMessageGroup ({ group, isLastGroup }: { group: ChatMessageGroup, isLastGroup: boolean }) {
  const enableDebug = /* !!me && */ !process.env.NEXT_PUBLIC_DISABLE_DEBUG_PANEL;
  const { engine_options } = useChatInfo(useCurrentChatController()) ?? {};

  const { params } = useChatPostState(useCurrentChatController());

  const [debugInfoOpen, setDebugInfoOpen] = useState(false);
  const [highlight, setHighlight] = useState(false);
  useEffect(() => {
    if (group.assistant && location.hash.slice(1) === String(group.assistant.id)) {
      setHighlight(true);
      document.getElementById(String(group.assistant.id))?.scrollIntoView({ behavior: 'instant', block: 'start' });
    }
  }, []);

  return (
    <section
      id={group.assistant && String(group.assistant.id)}
      className={cn('space-y-6 p-4 pt-12 border-b pb-10 last-of-type:border-b-0 last-of-type:border-pb-4', highlight && 'animate-highlight')}
      onAnimationEnd={() => setHighlight(false)}
    >
      <Collapsible open={debugInfoOpen} onOpenChange={setDebugInfoOpen}>
        <div className="relative pr-12">
          <h2 className="text-2xl font-normal whitespace-pre-wrap">{group.user.content}</h2>
          {enableDebug && <CollapsibleTrigger asChild>
            <Button className="absolute right-0 top-0 z-0 rounded-full" variant="ghost" size="sm">
              <InfoIcon className="h-4 w-4" />
              <span className="sr-only">Toggle</span>
            </Button>
          </CollapsibleTrigger>}
        </div>
        <CollapsibleContent>
          <DebugInfo group={group} />
        </CollapsibleContent>
      </Collapsible>

      {group.assistant?.version === 'Legacy' && <MessageAnnotationHistory message={group.assistant} />}
      {group.assistant?.version === 'StackVM' && <StackVMMessageAnnotationHistory message={group.assistant} />}

      {!engine_options?.hide_sources && <MessageSection className="!mt-1" message={group.assistant}>
        <MessageContextSources message={group.assistant} />
      </MessageSection>}

      <MessageSection className="space-y-2" message={group.assistant}>
        <MessageAnswer message={group.assistant} showBetaAlert={group.hasFirstAssistantMessage} />
        {group.assistant && <MessageAutoScroll message={group.assistant} />}
      </MessageSection>

      {group.assistant && <MessageError message={group.assistant} />}

      {group.assistant && <MessageOperations message={group.assistant} />}

      <MessageVerify assistant={group.assistant} />

      {!params && isLastGroup && group.hasLastAssistantMessage && <MessageRecommendQuestions assistant={group.assistant} />}
    </section>
  );
}
