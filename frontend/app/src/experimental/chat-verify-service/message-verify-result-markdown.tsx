import { rehypeHighlightOptions } from '@/components/remark-content/highlight';
import { cn } from '@/lib/utils';
import { capitalCase } from 'change-case-all';
import { h } from 'hastscript';
import { CheckIcon, XIcon } from 'lucide-react';
import type { Node, Parent, Root } from 'mdast';
import type { ReactNode } from 'react';
import * as jsxRuntime from 'react/jsx-runtime';
import rehypeHighlight from 'rehype-highlight';
import rehypeReact from 'rehype-react';
import remarkDirective from 'remark-directive';
import remarkGfm from 'remark-gfm';
import remarkParse from 'remark-parse';
import remarkRehype from 'remark-rehype';
import { format } from 'sql-formatter';
import { unified } from 'unified';
import { visit } from 'unist-util-visit';
import '@/components/remark-content/style.scss';

export function MessageVerifyResultMarkdown ({ content }: { content: string }) {
  return (
    <article className="remark-content prose prose-sm dark:prose-invert p-4" style={{ maxWidth: 'inherit' }}>
      {processor.processSync(content).result}
    </article>
  );
}

const processor = unified()
  .use(remarkParse)
  .use(remarkGfm)
  .use(formatSqlPlugin)
  .use(remarkDirective)
  .use(handleDirectivePlugin)
  .use(remarkRehype)
  .use(rehypeHighlight, rehypeHighlightOptions)
  .use(rehypeReact, {
    ...jsxRuntime as any,
    components: {
      MessageVerifyResult: ({ children, label, success }: { children: ReactNode, label: string, success: boolean }) => {
        return (
          <div className={cn('not-prose whitespace-break-spaces p-2 rounded text-xs', success ? 'bg-success/10' : 'bg-destructive/10')}>
            <div className={cn('mb-2 flex gap-1 items-center', success ? 'text-success' : 'text-destructive')}>
              {success ? <CheckIcon className="size-3" /> : <XIcon className="size-3" />}
              <span>{label}</span>
            </div>
            {children}
          </div>
        );
      },
      Void: () => {
        return null;
      },
    },
  })
  .freeze();

declare module 'mdast' {
  interface BlockContentMap {
    containerDirective: ContainerDirective;
  }

  interface PhrasingContentMap {
    leafDirective: LeafDirective;
    textDirective: TextDirective;
  }
}

interface ContainerDirective extends Parent {
  type: 'containerDirective';
  name: string;
  attributes: Record<string, string>;
}

interface LeafDirective extends Node {
  type: 'leafDirective';
  name: string;
  attributes: Record<string, string>;
}

interface TextDirective extends Node {
  type: 'textDirective';
  name: string;
  attributes: Record<string, string>;
}

function formatSqlPlugin () {
  return function (tree: Root) {
    visit(tree, function (node) {
      if (node.type === 'code') {
        if (node.lang === 'sql') {
          try {
            node.value = format(node.value, { language: 'mysql' });
          } catch {
          }
        }
      }
    });
  };
}

function handleDirectivePlugin () {
  return function (tree: Root) {
    visit(tree, function (node) {
      if (
        node.type === 'containerDirective'
      ) {
        const data = node.data || (node.data = {});
        const hast = h(node.name, { ...node.attributes, label: capitalCase(node.name), success: node.name === 'success' });

        data.hName = 'MessageVerifyResult';
        data.hProperties = hast.properties;
      }
      if (
        node.type === 'leafDirective' ||
        node.type === 'textDirective'
      ) {
        const data = node.data || (node.data = {});
        data.hName = 'Void';
        data.hProperties = {};
      }
    });
  };
}
