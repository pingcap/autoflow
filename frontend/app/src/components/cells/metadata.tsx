import { ScrollArea, ScrollBar } from '@/components/ui/scroll-area';
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip';
import type { CellContext } from '@tanstack/react-table';
import { BracesIcon } from 'lucide-react';
import { collapseAllNested, defaultStyles, JsonView } from 'react-json-view-lite';
import 'react-json-view-lite/dist/index.css';

export const metadataCell = (props: CellContext<any, any>) => {
  const metadata = props.getValue();

  if (!metadata) {
    return '-';
  }

  return (
    <>
      <Tooltip>
        <TooltipTrigger>
          <BracesIcon className="w-4 h-4" />
        </TooltipTrigger>
        <TooltipContent side="bottom">
          <ScrollArea className="max-w-[400px] max-h-[300px]">
            <JsonView data={metadata} shouldExpandNode={collapseAllNested} style={defaultStyles} />
            <ScrollBar orientation="vertical" />
            <ScrollBar orientation="horizontal" />
          </ScrollArea>
        </TooltipContent>
      </Tooltip>
    </>
  );
};
