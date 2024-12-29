import { MessageContextSourceCard } from '@/components/chat/message-content-sources';
import { CopyButton } from '@/components/copy-button';
import { RemarkContentContext } from '@/components/remark-content/context';
import { HoverCard, HoverCardContent, HoverCardTrigger } from '@/components/ui/hover-card';
import { cn } from '@/lib/utils';
import { HoverCardArrow, HoverCardPortal } from '@radix-ui/react-hover-card';
import { cloneElement, useContext, useState } from 'react';
import { isElement, isFragment } from 'react-is';
import * as jsxRuntime from 'react/jsx-runtime';
import { Options as RehypeReactOptions } from 'rehype-react';

function dirtyRewrite (some: any, id: string): any {
  if (some == null) return some;
  if (typeof some !== 'object') return some;

  if (isElement(some) || isFragment(some)) {
    const props = some.props as any;
    return cloneElement(some, {
      ...props,
      ...props.id ? { id: `${id}--${props.id}` } : {},
      children: dirtyRewrite(props.children, id),
    });
  }

  if (some instanceof Array) {
    return some.map(item => dirtyRewrite(item, id));
  }

  return some;
}

export const getRehypeReactOptions = ({ portalContainer }: { portalContainer: HTMLElement | undefined }): RehypeReactOptions => ({
  Fragment: (jsxRuntime as any).Fragment,
  jsx: (jsxRuntime as any).jsx,
  jsxs: (jsxRuntime as any).jsxs,
  passNode: true,
  components: {
    section ({ ...props }) {
      // eslint-disable-next-line react-hooks/rules-of-hooks
      const { reactId } = useContext(RemarkContentContext);

      if (!(props as any)['data-footnotes']) return <section {...props} />;
      return (
        <section {...props} className={cn(props.className /*, 'sr-only'*/)}>
          {dirtyRewrite(props.children, reactId)}
        </section>
      );
    },
    a ({ ...props }) {
      // eslint-disable-next-line react-hooks/rules-of-hooks
      const { reactId } = useContext(RemarkContentContext);

      // eslint-disable-next-line react-hooks/rules-of-hooks
      const [link, setLink] = useState<{ title: string, href: string | false }>();

      if (!(props as any)['data-footnote-ref']) return <a {...props} target="_blank" />;

      return (
        <HoverCard openDelay={0} onOpenChange={open => {
          if (open) {
            const id = props.href?.replace(/^#/, '');
            if (id) {
              const li = document.getElementById(reactId + '--' + id);
              if (li) {
                const a = li.querySelector(`a:first-child:not([data-footnote-backref])`) as HTMLAnchorElement | null;
                if (a) {
                  setLink({ title: a.textContent ?? a.href, href: a.href });
                  return;
                } else {
                  const text = li.querySelector('p')?.childNodes?.item(0)?.textContent;
                  if (text) {
                    setLink({ title: text, href: false });
                    return;
                  }
                }
              }
            }
            setLink(undefined);
          }
        }}>
          <HoverCardTrigger asChild>
            <a
              {...props}
              className={cn(props.className, 'cursor-default')}
              href={undefined}
              onClick={event => {
                event.preventDefault();
                event.stopPropagation();
              }}
            />
          </HoverCardTrigger>
          <HoverCardPortal container={portalContainer}>
            <HoverCardContent onPointerDownOutside={e => e.preventDefault()} className="p-1 w-[200px] overflow-hidden rounded-lg border text-xs">
              <HoverCardArrow className="fill-border" />
              {link
                ? link.href
                  ? <MessageContextSourceCard title={link?.title} href={link?.href} />
                  : link.title
                : null}
            </HoverCardContent>
          </HoverCardPortal>
        </HoverCard>
      );
    },
    pre ({ children, node, ...props }) {
      // eslint-disable-next-line react-hooks/rules-of-hooks
      const { rawContent } = useContext(RemarkContentContext);

      let isCodeBlock = false;
      let range: [number, number] | undefined;
      const firstChild = node?.children[0];
      if (firstChild?.type === 'element' && firstChild.tagName === 'code') {
        isCodeBlock = true;
        if (firstChild.position && firstChild.position.start.offset && firstChild.position.end.offset) {
          range = [firstChild.position.start.offset, firstChild.position.end.offset];
        }
      }

      return (
        <pre {...props}>
          {children}
          {isCodeBlock && <div className="absolute right-1 top-1 transition-opacity opacity-30 hover:opacity-100" data-role="codeblock-addon">
            {range && <CopyButton text={() => parseCode(rawContent, range)} />}
          </div>}
        </pre>
      );
    },
  },
});

function parseCode (raw: string, range: [number, number]) {
  // Unindent prefix tabs?
  return raw.slice(...range)
    .replace(/^\s*```[^\n]*\n/, '')
    .replace(/\n[^\n]*```$/, '');
}
