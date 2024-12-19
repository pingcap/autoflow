import type { FormControlWidgetProps } from '@/components/form/control-widget';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';
import { useId } from 'react';

export interface FileInputProps extends FormControlWidgetProps {
  accept: string[];
}

export const FileInput = ({
  name,
  accept,
  value,
  onChange,
  disabled,
  ref,
  onBlur,
  ...props
}: FileInputProps) => {
  const id = useId();
  return (
    <>
      <input
        className="hidden"
        id={id}
        name={name}
        type="file"
        accept={accept.join(', ')}
        onChange={event => {
          const file = event.target.files?.item(0) ?? undefined;
          onChange?.(file);
        }}
        disabled={disabled}
      />
      <Button
        variant="outline"
        disabled={disabled}
        ref={ref}
        onBlur={onBlur}
        {...props}
        className={cn('flex w-full justify-start font-normal', value == null && 'text-muted-foreground')}
        onClick={(event) => {
          (props as any).onClick?.(event);
          if (!event.defaultPrevented) {
            document.getElementById(id)?.click();
          }
        }}
        type="button"
      >
        {value
          ? value.name
          : 'Select file'}
      </Button>
    </>
  );
};
