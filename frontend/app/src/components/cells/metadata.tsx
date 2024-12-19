import { ThemedStyle } from '@/components/themed-style';
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover';
import type { CellContext } from '@tanstack/react-table';
import JsonView from '@uiw/react-json-view';
import { darkTheme } from '@uiw/react-json-view/dark';
import { lightTheme } from '@uiw/react-json-view/light';
import { AlignLeftIcon, BracesIcon, BracketsIcon } from 'lucide-react';
import type { ReactElement, ReactNode } from 'react';

export const metadataCell = (props: CellContext<any, any>) => {
  const metadata = props.getValue();

  if (metadata == null) {
    return <pre className="text-xs">(nul)</pre>;
  }

  let icon: ReactElement | null;
  let text: ReactNode;

  if (typeof metadata === 'object') {
    if (metadata instanceof Array) {
      icon = <BracketsIcon className="w-4 h-4" />;
      text = <span className="text-muted-foreground">{`${metadata.length} items`}</span>;
    } else {
      icon = <BracesIcon className="w-4 h-4" />;
      text = <span className="text-muted-foreground">{`${Object.keys(metadata).length} keys`}</span>;
    }
  } else {
    const stringValue = String(metadata);
    if (stringValue.length < 25) {
      return stringValue;
    }
    icon = <AlignLeftIcon className="w-4 h-4" />;
    text = <span className="text-muted-foreground">{typeof metadata}</span>;
  }

  return (
    <Popover modal>
      <PopoverTrigger className="inline-flex gap-1 items-center">
        {icon} {text}
      </PopoverTrigger>
      <PopoverContent side="bottom" className="w-96 max-h-72 overflow-auto scroll-smooth">
        <ThemedStyle dark={darkTheme} light={lightTheme}>
          <JsonView value={metadata} collapsed={2} />
        </ThemedStyle>
      </PopoverContent>
    </Popover>
  );
};
