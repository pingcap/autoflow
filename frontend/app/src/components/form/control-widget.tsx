import { Checkbox } from '@/components/ui/checkbox';
import { Command, CommandEmpty, CommandGroup, CommandInput, CommandItem, CommandList } from '@/components/ui/command';
import { Popover, PopoverContent } from '@/components/ui/popover';
import { Switch } from '@/components/ui/switch';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';
import { getErrorMessage } from '@/lib/errors';
import type { KeyOfType } from '@/lib/typing-utils';
import { cn } from '@/lib/utils';
import * as PopoverPrimitive from '@radix-ui/react-popover';
import type { SwitchProps } from '@radix-ui/react-switch';
import type { DeepKeys } from '@tanstack/react-form';
import { CheckIcon, ChevronDown, Loader2Icon, TriangleAlertIcon, XCircleIcon } from 'lucide-react';
import * as React from 'react';
import { type ChangeEvent, type ComponentProps, type FC, forwardRef, type Key, type ReactElement, type ReactNode, type Ref, useState } from 'react';

export interface FormControlWidgetProps<T, Optional extends boolean = false> {
  id?: string;
  'aria-describedby'?: string;
  'aria-invalid'?: boolean;

  onChange?: (value: (Optional extends false ? T : T | undefined) | ChangeEvent<any>) => void;
  onBlur?: () => void;
  value?: T;
  disabled?: boolean;
  name?: any; // type issue
  ref?: Ref<any>;
}

export { Input as FormInput, type InputProps as FormInputProps } from '@/components/ui/input';

export { Textarea as FormTextarea, type TextareaProps as FormTextareaProps } from '@/components/ui/textarea';

export interface FormSwitchProps extends FormControlWidgetProps<boolean>, Omit<SwitchProps, 'checked' | 'onCheckedChange' | keyof FormControlWidgetProps<boolean>> {
}

export const FormSwitch = forwardRef<any, FormSwitchProps>(({ value, onChange, ...props }, forwardedRef) => {
  return (
    <Switch
      {...props}
      ref={forwardedRef}
      checked={value}
      onCheckedChange={onChange}
    />
  );
});

FormSwitch.displayName = 'FormSwitch';

export interface FormCheckboxProps extends FormControlWidgetProps<boolean>, Omit<ComponentProps<typeof Checkbox>, 'checked' | 'onCheckedChange' | keyof FormControlWidgetProps<boolean>> {
}

export const FormCheckbox = forwardRef<any, FormCheckboxProps>(({ value, onChange, ...props }, forwardedRef) => {
  return (
    <Checkbox
      {...props}
      ref={forwardedRef}
      checked={value}
      onCheckedChange={value => onChange?.(!!value)}
    />
  );
});

FormCheckbox.displayName = 'FormCheckbox';

export interface FormSelectConfig<T extends object, K extends KeyOfType<T, Key>> {
  loading?: boolean;
  error?: unknown;
  options: T[];
  key: K;
  clearable?: boolean;
  itemClassName?: string;
  renderOption: (option: T) => ReactNode;
  renderValue?: (option: T) => ReactNode;
}

export interface FormComboboxConfig<T extends object, K extends KeyOfType<T, Key>> extends FormSelectConfig<T, K> {
  optionKeywords: (option: T) => string[];
  renderCreateOption?: (wrapper: FC<{ onSelect: () => void, children: ReactNode }>, onCreated: (item: T) => void) => ReactNode;
}

export interface FormComboboxProps<T extends object, K extends KeyOfType<T, Key>> extends FormControlWidgetProps<T[K], true> {
  children?: ReactElement<any>;
  placeholder?: string;
  config: FormComboboxConfig<T, K>;
  contentWidth?: 'anchor';
  ref?: Ref<any>;
}

export function FormCombobox<T extends object, K extends KeyOfType<T, Key>> ({ ref, config, placeholder, value, onChange, name, disabled, children, contentWidth = 'anchor', ...props }: FormComboboxProps<T, K>) {
  const [open, setOpen] = useState(false);
  const isConfigReady = !config.loading && !config.error;
  const current = config.options.find(option => option[config.key] === value);

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <div className={cn('flex items-center gap-2', (props as any).className)}>
        <PopoverPrimitive.Trigger
          ref={ref}
          disabled={disabled || !isConfigReady}
          {...props}
          className={cn(
            'flex h-10 w-full items-center justify-between rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50 [&>span]:line-clamp-1',
          )}
          asChild={!!children}
        >
          {config.loading
            ? <span>Loading options...</span>
            : !!config.error
              ? <span className="text-destructive">{getErrorMessage(config.error)}</span>
              : (children ? children : current ? (config.renderValue ?? config.renderOption)(current) : <span className="text-muted-foreground">{placeholder}</span>)
          }
          <span className="flex-1" />
          {config.loading
            ? <Loader2Icon className="size-4 opacity-50 animate-spin repeat-infinite" />
            : config.error
              ? <TriangleAlertIcon className="size-4 text-destructive opacity-50" />
              : config.clearable !== false && current != null && !disabled
                ? <FormComboboxClearButton onClick={() => onChange?.(undefined)} />
                : <ChevronDown className="h-4 w-4 opacity-50" />}
        </PopoverPrimitive.Trigger>
      </div>
      <PopoverContent className={cn('p-0 focus:outline-none', contentWidth === 'anchor' && 'w-[--radix-popover-trigger-width]')} align="start" collisionPadding={8}>
        <Command>
          <CommandInput />
          <CommandList>
            <CommandGroup>
              {config.renderCreateOption && config.renderCreateOption(
                ({ onSelect, children }) => (
                  <CommandItem value="$create$" onSelect={onSelect} className={config.itemClassName} forceMount>
                    {children}
                  </CommandItem>
                ),
                (item) => {
                  onChange?.(item[config.key]);
                  setOpen(false);
                })}
              {config.options.map(option => (
                <CommandItem
                  key={option[config.key] as Key}
                  value={String(option[config.key])}
                  keywords={config.optionKeywords(option).flatMap(item => item.split(/\s+/))}
                  className={cn('group', config.itemClassName)}
                  onSelect={value => {
                    const item = config.options.find(option => String(option[config.key]) === value);
                    if (item) {
                      onChange?.(item[config.key]);
                      setOpen(false);
                    }
                  }}
                >
                  {config.renderOption(option)}
                  <CheckIcon className={cn('ml-auto size-4 opacity-0', current?.[config.key] === option[config.key] && 'opacity-100')} />
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

FormCombobox.displayName = 'FormCombobox';

function FormComboboxClearButton ({ onClick }: { onClick?: () => void }) {
  return (
    <TooltipProvider delayDuration={0}>
      <Tooltip>
        <TooltipTrigger asChild>
          <span role="button" className="ml-2 opacity-50 hover:opacity-100" onClick={onClick}>
            <XCircleIcon className="size-4" />
          </span>
        </TooltipTrigger>
        <TooltipContent>
          Clear select
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}
