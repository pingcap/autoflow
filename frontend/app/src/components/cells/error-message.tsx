import { PythonViewer } from '@/components/py-viewer';
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
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
    <Dialog>
      <DialogTrigger>
        {shortcut}{'... '}
        <span className="text-muted-foreground">
          ({children.length + ' characters'})
        </span>
      </DialogTrigger>
      <DialogContent className="max-w-screen-lg h-[80vh]">
        <DialogHeader>
          <DialogTitle>
            Error Message
          </DialogTitle>
          <DialogDescription className="sr-only" />
        </DialogHeader>
        <div className="size-full overflow-scroll">
          <PythonViewer value={children} />
        </div>
      </DialogContent>
    </Dialog>
  );
}
