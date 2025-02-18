import { type FormControlWidgetProps } from '@/components/form/control-widget';
import { useAllKnowledgeBases } from '@/components/knowledge-base/hooks';
import { Command, CommandEmpty, CommandGroup, CommandInput, CommandItem, CommandList } from '@/components/ui/command';
import { Popover, PopoverContent } from '@/components/ui/popover';
import { getErrorMessage } from '@/lib/errors';
import { cn } from '@/lib/utils';
import * as PopoverPrimitive from '@radix-ui/react-popover';
import { AlertTriangleIcon, CheckIcon, DotIcon } from 'lucide-react';
import * as React from 'react';
import { useState } from 'react';

export function KBListSelect ({ ref, disabled, value, onChange, ...props }: FormControlWidgetProps<number[]>) {
  const [open, setOpen] = useState(false);
  const { data: knowledgeBases, isLoading, error } = useAllKnowledgeBases();
  const isConfigReady = !isLoading && !error;

  const current = value?.map(id => knowledgeBases?.find(kb => kb.id === id));

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <div className={cn('flex items-center gap-2')}>
        <PopoverPrimitive.Trigger
          ref={ref}
          disabled={disabled || !isConfigReady}
          className={cn(
            'flex flex-col min-h-10 w-full text-left items-stretch justify-start rounded-md border border-input bg-background px-3 py-1 text-sm ring-offset-background placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50',
          )}
          {...props}
        >
          {isLoading
            ? <span>Loading options...</span>
            : !!error
              ? <span className="text-destructive">{getErrorMessage(error)}</span>
              : !!current?.length
                ? current.map((option, index) => (
                  option ? (
                    <div key={option.id} className="w-full block border-t first-of-type:border-t-0 py-2">
                      <span>{option.name}</span>
                      <div className="text-xs text-muted-foreground ml-2 inline-flex gap-1 items-center">
                      <span>
                        {(option.documents_total ?? 0) || <><AlertTriangleIcon className="text-warning inline-flex size-3 mr-0.5" /> no</>} documents
                      </span>
                        <DotIcon className="size-4" />
                        <span className="text-xs text-muted-foreground">
                        {(option.data_sources_total ?? 0) || <><AlertTriangleIcon className="inline-flex size-3 mr-0.5" /> no</>} data sources
                      </span>
                      </div>
                    </div>
                  ) : <span key={value?.[index]}>UNKNOWN KNOWLEDGE BASE {value?.[index]}</span>
                )) : <span className="pt-1 text-muted-foreground">Select Knowledge Bases</span>
          }
        </PopoverPrimitive.Trigger>
      </div>
      <PopoverContent className={cn('p-0 focus:outline-none w-[--radix-popover-trigger-width]')} align="start" collisionPadding={8}>
        <Command>
          <CommandInput />
          <CommandList>
            <CommandGroup>
              {knowledgeBases?.map(option => (
                <CommandItem
                  key={option.id}
                  value={String(option.id)}
                  keywords={[option.name, option.description ?? '']}
                  className={cn('group')}
                  onSelect={idValue => {
                    const id = knowledgeBases?.find(option => String(option.id) === idValue)?.id;
                    if (id) {
                      if (value?.includes(id)) {
                        onChange?.(value.filter(v => v !== id));
                      } else {
                        onChange?.([...(value ?? []), id]);
                      }
                    }
                  }}
                >
                  <div className="space-y-1">
                    <div>
                      <strong>
                        {option.name}
                      </strong>
                    </div>
                    <div className="text-xs text-muted-foreground flex gap-1 items-center">
                      <span>
                        {(option.documents_total ?? 0) || <><AlertTriangleIcon className="text-warning inline-flex size-3 mr-0.5" /> no</>} documents
                      </span>
                      <DotIcon className="size-4" />
                      <span>
                        {(option.data_sources_total ?? 0) || <><AlertTriangleIcon className="inline-flex size-3 mr-0.5" /> no</>} data sources
                      </span>
                    </div>
                    <div className="text-xs text-muted-foreground">
                      {option.description}
                    </div>
                  </div>
                  <CheckIcon className={cn('ml-auto size-4 opacity-0 flex-shrink-0', value?.includes(option.id) && 'opacity-100')} />
                </CommandItem>
              ))}
            </CommandGroup>
            <CommandEmpty className="text-muted-foreground/50 text-xs p-4 text-center">
              Empty List
            </CommandEmpty>
          </CommandList>
        </Command>
      </PopoverContent>
    </Popover>
  );
}

export function KBListSelectForObjectValue ({ value, onChange, ...props }: FormControlWidgetProps<{ id: number }[], true>) {
  return (
    <KBListSelect
      value={value?.map(v => v.id) ?? []}
      onChange={value => {
        onChange?.((value as number[]).map(id => ({ id })));
      }}
      {...props}
    />
  );
}
