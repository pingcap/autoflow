import { type FormControlWidgetProps, FormTextarea } from '@/components/form/control-widget';
import { buttonVariants } from '@/components/ui/button';
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { cn } from '@/lib/utils';
import { forwardRef } from 'react';

export interface PromptInputProps extends FormControlWidgetProps {
  className?: string;
}

export const PromptInput = forwardRef<any, PromptInputProps>(({ className, ...props }: PromptInputProps, ref) => {
  return (
    <Dialog>
      <DialogTrigger ref={ref} className={cn(buttonVariants({ variant: 'outline' }), 'flex gap-1 w-full font-normal', className)}>
        {'Edit prompt'}
        <span className="text-muted-foreground">({props.value.length} characters)</span>
      </DialogTrigger>
      <DialogContent className="h-2/3">
        <DialogHeader className="sr-only">
          <DialogTitle>Update Prompt</DialogTitle>
          <DialogDescription />
        </DialogHeader>
        <FormTextarea {...props} />
      </DialogContent>
    </Dialog>
  );
});

PromptInput.displayName = 'PromptInput';
