'use client';

import { Button, type ButtonProps } from '@/components/ui/button';
import { cn } from '@/lib/utils';
import copy from 'copy-to-clipboard';
import { ClipboardCheckIcon, ClipboardIcon } from 'lucide-react';
import { useLayoutEffect, useState } from 'react';

export interface CopyButtonProps extends Omit<ButtonProps, 'children' | 'type'> {
  text: string | (() => string);
  autoCopy?: boolean;
}

export function CopyButton ({ text, className, onClick, autoCopy, ...props }: CopyButtonProps) {
  const [copied, setCopied] = useState(false);

  useLayoutEffect(() => {
    setCopied(false);
    if (autoCopy) {
      setCopied(copy(typeof text === 'string' ? text : text()));
    }
  }, [text]);

  return (
    <Button
      size="icon"
      variant="ghost"
      className={cn('rounded-full w-7 h-7 transition-colors text-foreground', copied && 'text-success hover:text-success hover:bg-success/10')}
      onClick={(event) => {
        onClick?.(event);
        if (!event.defaultPrevented) {
          setCopied(copy(typeof text === 'string' ? text : text()));
        }
      }}
      {...props}
      type="button"
    >
      {copied
        ? <ClipboardCheckIcon className="w-4 h-4" />
        : <ClipboardIcon className="w-4 h-4" />}
    </Button>
  );
}