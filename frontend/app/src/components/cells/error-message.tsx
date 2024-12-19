import { HoverCard, HoverCardContent, HoverCardTrigger } from '@/components/ui/hover-card';
import type { CellContext } from '@tanstack/react-table';

export function errorMessageCell<Row> (trimLength = 25) {
  return function ErrorMessageCell (context: CellContext<Row, string | null | undefined>) {
    return <AutoErrorMessagePopper trimLength={trimLength}>{context.getValue() ?? '-'}</AutoErrorMessagePopper>;
  };
}

export function AutoErrorMessagePopper ({ trimLength = 25, children }: { trimLength?: number, children: string | null }) {
  if (!children || children.length <= trimLength) {
    return children;
  }

  const shortcut = children.slice(0, trimLength);

  return (
    <HoverCard>
      <HoverCardTrigger>
        {shortcut}{'... '}
        <span className="text-muted-foreground">
          ({children.length + ' characters'})
        </span>
      </HoverCardTrigger>
      <HoverCardContent className="w-96 h-48">
        <div className="size-full overflow-scroll">
          <pre className="whitespace-pre">{children}</pre>
        </div>
      </HoverCardContent>
    </HoverCard>
  );
}
