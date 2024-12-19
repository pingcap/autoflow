import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Button } from '@/components/ui/button';
import { Toaster } from '@/components/ui/sonner';
import { cn } from '@/lib/utils';
import type { Meta, StoryObj } from '@storybook/react';
import { MailQuestionIcon } from 'lucide-react';
import type { ComponentProps } from 'react';
import { toast } from 'sonner';

const meta = {
  title: 'Theme',
  render () {
    return (
      <div className="space-y-5">
        <section>
          <ColorExample name="Warning" className="text-warning" />
          <ColorExample name="Info" className="text-info" />
          <ColorExample name="Success" className="text-success" />
        </section>
        <section>
          <AlertExample name="default" />
          <AlertExample name="destructive" />
          <AlertExample name="warning" />
          <AlertExample name="info" />
          <AlertExample name="success" />
        </section>
        <section>
          <Toaster cn={cn} />
          <Button
            onClick={() => {
              toast.success('Success', {
                description: 'Description is here',
                icon: <MailQuestionIcon className="size-4" />,
              });
            }}>
            Success
          </Button>
          <Button
            onClick={() => {
              toast.warning('Warning', {
                description: 'Description is here',
                icon: <MailQuestionIcon className="size-4" />,
              });
            }}>
            Warning
          </Button>
          <Button
            onClick={() => {
              toast.info('Info', {
                description: 'Description is here',
                icon: <MailQuestionIcon className="size-4" />,
              });
            }}>
            Info
          </Button>
          <Button
            onClick={() => {
              toast.error('Error', {
                description: 'Description is here',
                icon: <MailQuestionIcon className="size-4" />,
              });
            }}>
            Error
          </Button>
        </section>
      </div>
    );
  },
} satisfies Meta;

export default meta;

export const Default = {} satisfies StoryObj<typeof meta>;

function ColorExample ({ name, className }: { name: string, className: string }) {
  return (
    <div className="space-y-2">
      <h6 className="font-semibold">{name}</h6>
      <div className="flex gap-4 items-center">
        <div className="bg-background p-4">
        <span className={cn('flex gap-2 items-center p-2 border border-current rounded', className)}>
          <span>
            {className}
          </span>
          <span className="bg-current size-5 rounded" />
        </span>
        </div>
        <div className="dark bg-background p-4 rounded-lg">
        <span className={cn('flex gap-2 items-center p-2 border border-current rounded', className)}>
          <span>
            {className}
          </span>
          <span className="bg-current size-5 rounded" />
        </span>
        </div>
      </div>
    </div>
  );
}

function AlertExample ({ name }: { name: ComponentProps<typeof Alert>['variant'] }) {
  return (
    <div className="space-y-2">
      <h6 className="font-semibold">{name}</h6>
      <div className="flex gap-4 items-center">
        <div className="bg-background p-4">
          <Alert variant={name}>
            <MailQuestionIcon />
            <AlertTitle>Title</AlertTitle>
            <AlertDescription>Description</AlertDescription>
          </Alert>
        </div>
        <div className="dark bg-background p-4 rounded-lg">
          <Alert variant={name}>
            <MailQuestionIcon />
            <AlertTitle>Title</AlertTitle>
            <AlertDescription>Description</AlertDescription>
          </Alert>
        </div>
      </div>
    </div>
  );
}
