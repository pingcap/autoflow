import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { PlusIcon } from 'lucide-react';
import { forwardRef } from 'react';
import type { ControllerRenderProps } from 'react-hook-form';

/**
 * @deprecated
 */
export const LinkArrayField = forwardRef<HTMLDivElement, ControllerRenderProps>(({ value, onChange, name, disabled, onBlur }, ref) => {
  return (
    <div className="space-y-1" ref={ref}>
      {(value as { text: string, href: string }[] | null)?.map((item, index) => (
        <div key={index} className="flex gap-2 items-center">
          <Input
            className="w-40"
            disabled={disabled}
            value={item.text}
            onChange={event => {
              value = [...value];
              value[index] = {
                ...value[index],
                text: event.target.value,
              };
              onChange(value);
            }}
            onBlur={onBlur}
          />
          <Input
            value={item.href}
            disabled={disabled}
            onChange={event => {
              value = [...value];
              value[index] = {
                ...value[index],
                href: event.target.value,
              };
              onChange(value);
            }}
            onBlur={onBlur}
          />
          <Button
            variant="secondary"
            disabled={disabled}
            type="button"
            onClick={() => {
              value = [...value];
              value.splice(index, 0, { text: '', href: '' });
              onChange(value);
            }}
          >
            Add
          </Button>
          <Button
            type="button"
            disabled={disabled}
            variant="ghost"
            onClick={() => {
              value = [...value];
              value.splice(index, 1);
              onChange(value);
            }}
          >
            Delete
          </Button>
        </div>
      ))}
      {!disabled && <Button
        className="gap-2"
        variant="ghost"
        type="button"
        onClick={() => onChange([...value, { text: '', href: '' }])}
      >
        <PlusIcon className="size-4" />
        Add
      </Button>}
    </div>
  );
});

LinkArrayField.displayName = 'LinkArrayField';
