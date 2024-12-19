'use client';

import { listChatEngines } from '@/api/chat-engines';
import { useAuth } from '@/components/auth/AuthProvider';
import { useAllChatEngines } from '@/components/chat-engine/hooks';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { cn } from '@/lib/utils';
import isHotkey from 'is-hotkey';
import { ArrowUpIcon } from 'lucide-react';
import { type ChangeEvent, type Ref, useCallback, useRef } from 'react';
import TextareaAutosize, { type TextareaAutosizeProps } from 'react-textarea-autosize';
import useSWR from 'swr';

export interface MessageInputProps {
  className?: string,
  disabled?: boolean,
  actionDisabled?: boolean,
  inputRef?: Ref<HTMLTextAreaElement>,
  inputProps?: TextareaAutosizeProps,
  engine?: string,
  onEngineChange?: (name: string) => void,
}

export function MessageInput ({
  className,
  disabled,
  actionDisabled,
  inputRef,
  inputProps,
  engine,
  onEngineChange,
}: MessageInputProps) {
  const auth = useAuth();
  const buttonRef = useRef<HTMLButtonElement>(null);

  const onChangeRef = useRef(inputProps?.onChange);
  onChangeRef.current = inputProps?.onChange;
  const handleChange = useCallback((ev: ChangeEvent<HTMLTextAreaElement>) => {
    onChangeRef.current?.(ev);
  }, []);

  const showShowSelectChatEngine = !!auth.me?.is_superuser && !!onEngineChange;
  const { data, isLoading } = useAllChatEngines();

  return (
    <div className={cn('bg-background border p-2 rounded-lg', className)}>
      <TextareaAutosize
        placeholder="Input your question here..."
        onKeyDown={e => {
          if (!e.nativeEvent.isComposing && isHotkey('mod+Enter', e) && !actionDisabled) {
            e.preventDefault();
            buttonRef.current?.click();
          }
        }}
        {...inputProps}
        onChange={handleChange}
        ref={inputRef}
        className="w-full border-none ring-0 outline-none bg-background focus-visible:ring-0 focus-visible:ring-offset-0 resize-none placeholder:font-light placeholder-gray-400 dark:placeholder-gray-500 max-h-72"
        disabled={disabled || inputProps?.disabled}
        minRows={4}
      />
      <div className="flex items-center justify-end gap-2">
        {showShowSelectChatEngine && <Select value={engine ?? ''} onValueChange={value => onEngineChange?.(value)}>
          <SelectTrigger className="w-max border-none h-max" disabled={isLoading}>
            <SelectValue placeholder="Select Chat Engine" />
          </SelectTrigger>
          <SelectContent>
            {data?.map(item => (
              <SelectItem key={item.name} value={String(item.name)} textValue={item.name}>
              <span className="flex items-center gap-2">
                {item.is_default ? <Badge variant="outline" className="text-success border-success/70">default</Badge> : item.name}
                {!!item.engine_options.external_engine_config?.stream_chat_api_url
                  ? <Badge>External Engine (StackVM)</Badge>
                  : item.engine_options.knowledge_graph?.enabled !== false /* TODO: require default config */
                    ? <Badge variant="secondary">Knowledge graph enabled</Badge>
                    : undefined}
              </span>
              </SelectItem>
            ))}
          </SelectContent>
        </Select>}
        <Button size="icon" className="rounded-full flex-shrink-0 w-8 h-8 p-2" disabled={actionDisabled || disabled} ref={buttonRef}>
          <ArrowUpIcon className="w-full h-full" />
        </Button>
      </div>
    </div>
  );
}
