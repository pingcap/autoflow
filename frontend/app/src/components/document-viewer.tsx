import { PromptViewer } from '@/components/chat-engine/prompt-viewer';
import { Button } from '@/components/ui/button';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { ScrollArea } from '@/components/ui/scroll-area';
import { useEffect, useState } from 'react';

import Highlight from 'highlight.js/lib/core';
import markdown from 'highlight.js/lib/languages/markdown';
import './code-theme.scss';

export interface DocumentPreviewProps {
  content: string;
  mime: string;
}

Highlight.registerLanguage('markdown', markdown);

export function DocumentViewer ({ content, mime }: DocumentPreviewProps) {
  if (mime === 'text/markdown') {
    return <MarkdownViewer value={content} />;
  } else {
    return (
      <div className="whitespace-pre-wrap text-xs font-mono">
        {content}
      </div>
    );
  }
}

const nf = new Intl.NumberFormat('en-US');

export function DocumentPreviewDialog ({ title, name, mime, content }: { title: string, name: string, mime: string, content: string }) {
  return (
    <Dialog>
      <DialogTrigger asChild>
        <Button className='text-xs p-2' variant="ghost" size="sm">
          {name} ({nf.format(content.length)} characters)
        </Button>
      </DialogTrigger>
      <DialogContent className="max-w-[720px] w-full">
        <DialogHeader>
          <DialogTitle>
            {title}
          </DialogTitle>
        </DialogHeader>
        <ScrollArea className="h-[80vh]">
          <DocumentViewer mime={mime} content={content} />
        </ScrollArea>
      </DialogContent>
    </Dialog>
  );
}

function MarkdownViewer ({ value: propValue }: { value: string }) {
  const [value, setValue] = useState(propValue);

  useEffect(() => {
    setValue(propValue);
    try {
      const { value: result } = Highlight.highlight('markdown', propValue);
      setValue(result);
    } catch {
    }
  }, [propValue]);

  return (
    <code>
      <pre className="whitespace-pre-wrap text-xs font-mono" dangerouslySetInnerHTML={{ __html: value }} />
    </code>
  );
}
