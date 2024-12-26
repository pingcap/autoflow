import { cn } from '@/lib/utils';
import * as ProgressPrimitive from '@radix-ui/react-progress';
import type { CellContext } from '@tanstack/react-table';
import * as React from 'react';
import { type CSSProperties } from 'react';

interface PercentCellConfig {
  multiply?: number;
  colorStops?: {
    checkpoint: number
    color: string
  }[];
}

export const percent = (
  cell: CellContext<any, number | undefined | null>,
  { multiply = 100, colorStops }: PercentCellConfig = {},
) => (
  <span className="flex items-center gap-1">
    <ProgressPrimitive.Root
      className={cn(
        'relative overflow-hidden rounded-full bg-secondary flex-shrink-0 w-16 h-2',
      )}
    >
      <ProgressPrimitive.Indicator
        className="h-full w-full flex-1 bg-primary transition-all"
        style={{
          transform: `translateX(-${100 - ((cell.getValue() ?? 0) * multiply)}%)`,
          ...getStyle(cell.getValue() ?? 0, colorStops),
        }}
      />
    </ProgressPrimitive.Root>
    <code className="text-xs">
      {cell.getValue()?.toFixed(2) ?? '-'}
    </code>
  </span>
);

function getStyle (value: number, stops: {
  checkpoint: number
  color: string
}[] | undefined): CSSProperties {
  if (!stops || stops.length === 0) {
    return {};
  }
  let from = stops.findLast(stop => stop.checkpoint <= value)!;
  let to = stops.find(stop => stop.checkpoint >= value)!;

  if (!from || !to) {
    return {};
  }

  if (from.checkpoint === to.checkpoint) {
    return {
      backgroundColor: from.color,
    };
  }

  const p1 = (value - from.checkpoint) / (to.checkpoint - from.checkpoint) * 100;

  return {
    backgroundColor: `color-mix(in srgb, ${from.color} ${100 - p1}%, ${to.color} ${p1}%)`,
  };
}
