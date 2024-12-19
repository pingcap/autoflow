import type { CellContext } from '@tanstack/react-table';
import { CheckIcon, XIcon } from 'lucide-react';

export function boolean (props: CellContext<any, boolean | undefined | null>) {
  const bool = props.getValue();

  if (bool == null) {
    return <span className="text-muted-foreground">-</span>;
  }

  if (bool) {
    return (
      <span className="text-success inline-flex gap-1 items-center">
        <CheckIcon className="size-4" />
        Yes
      </span>
    );
  } else {
    return (
      <span className="text-muted-foreground inline-flex gap-1 items-center">
        <XIcon className="size-4" />
        No
      </span>
    );
  }
}
