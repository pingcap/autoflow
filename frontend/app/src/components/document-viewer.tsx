import './code-theme.scss';

import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { useEffect, useState } from 'react';

import { Button } from '@/components/ui/button';
import Highlight from 'highlight.js/lib/core';
import { ScrollArea } from '@/components/ui/scroll-area';
import markdown from 'highlight.js/lib/languages/markdown';

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
        <Button className="text-xs p-2 font-normal font-mono" variant="ghost" size="sm">
          {name} <span className="text-muted-foreground">({nf.format(content.length)} characters)</span>
        </Button>
      </DialogTrigger>
      <DialogContent className="max-w-[720px] w-full">
        <DialogHeader>
          <DialogTitle>
            {title}
          </DialogTitle>
          <DialogDescription className="sr-only" />
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
      const { value: result } = Highlight.highlight(propValue, { language: 'markdown' });
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
