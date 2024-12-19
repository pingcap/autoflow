import { DangerousActionButton, type DangerousActionButtonProps } from '@/components/dangerous-action-button';
import { buttonVariants } from '@/components/ui/button';
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from '@/components/ui/dropdown-menu';
import { useDataTable } from '@/components/use-data-table';
import { cn } from '@/lib/utils';
import type { CellContext } from '@tanstack/react-table';
import { EllipsisVerticalIcon, Loader2Icon } from 'lucide-react';
import type { AppRouterInstance } from 'next/dist/shared/lib/app-router-context.shared-runtime';
import { useRouter } from 'next/navigation';
import { type Dispatch, type ReactNode, type SetStateAction, type TransitionStartFunction, useState, useTransition } from 'react';

export interface CellAction {
  key?: string | number;
  icon?: ReactNode;
  title?: ReactNode;
  disabled?: boolean;
  dangerous?: Pick<DangerousActionButtonProps, 'dialogDescription' | 'dialogTitle'>;
  action: (context: ActionUIContext) => Promise<void> | void;
}

export interface ActionUIContext {
  table: ReturnType<typeof useDataTable>;
  startTransition: TransitionStartFunction;
  router: AppRouterInstance;
  dropdownOpen: boolean;
  setDropdownOpen: Dispatch<SetStateAction<boolean>>;
}

export function actions<Row> (items: (row: Row) => CellAction[]) {
  // eslint-disable-next-line react/display-name
  return (props: CellContext<Row, any>) => {
    const [open, setOpen] = useState(false);
    const actionItems = items(props.row.original);

    return (
      <DropdownMenu open={open} onOpenChange={setOpen}>
        <DropdownMenuTrigger className={cn(buttonVariants({ variant: 'ghost', size: 'icon' }), 'text-muted-foreground p-1 size-6')} disabled={actionItems.length === 0}>
          <EllipsisVerticalIcon className="size-4" />
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end">
          {actionItems.map((item, index) => (
            <Action key={item.key ?? index} item={item} open={open} setOpen={setOpen} />
          ))}
        </DropdownMenuContent>
      </DropdownMenu>
    );
  };
}

function Action ({ item, open, setOpen }: { item: CellAction, open: boolean, setOpen: Dispatch<SetStateAction<boolean>> }) {
  const table = useDataTable();
  const [busy, setBusy] = useState(false);
  const [transitioning, startTransition] = useTransition();
  const router = useRouter();

  const onAction = async () => {
    try {
      setBusy(true);
      await item.action({ startTransition, router, table, dropdownOpen: open, setDropdownOpen: setOpen });
    } finally {
      setBusy(false);
    }
  };

  let el = (
    <DropdownMenuItem
      className={cn('gap-2 cursor-pointer disabled:cursor-not-allowed text-xs', item.dangerous && 'text-destructive focus:bg-destructive/10 focus:text-destructive')}
      disabled={item.disabled || transitioning}
      onSelect={item.dangerous
        ? (event) => {
          event.preventDefault();
        }
        : (event) => {
          event.preventDefault();
          void onAction();
        }}
    >
      {item.icon ? (busy || transitioning) ? <Loader2Icon className="size-3 animate-spin repeat-infinite" /> : item.icon : null}
      {item.title}
    </DropdownMenuItem>
  );

  if (item.dangerous) {
    el = (
      <DangerousActionButton {...item.dangerous} action={onAction} asChild>
        {el}
      </DangerousActionButton>
    );
  }

  return el;
}
