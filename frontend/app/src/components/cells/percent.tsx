import { Progress } from '@/components/ui/progress';
import type { CellContext } from '@tanstack/react-table';

export const percent = (cell: CellContext<any, number | undefined | null>, multiply = 100) => (
  <span className="flex items-center gap-1">
    <Progress
      className="flex-shrink-0 w-16 h-2"
      value={(cell.getValue() ?? 0) * multiply}
    />
    <code className="text-xs">
      {cell.getValue()?.toFixed(2) ?? '-'}
    </code>
  </span>
);
