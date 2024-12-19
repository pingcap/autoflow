import type { FeedbackParams } from '@/api/chats';
import { usePortalContainer } from '@/components/portal-provider';
import { Button } from '@/components/ui/button';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Textarea } from '@/components/ui/textarea';
import { ToggleGroup, ToggleGroupItem } from '@/components/ui/toggle-group';
import { Loader2Icon, ThumbsDownIcon, ThumbsUpIcon } from 'lucide-react';
import { type ReactNode, useEffect, useState } from 'react';

export function MessageFeedback ({ initial, onFeedback, defaultAction, children }: { initial?: FeedbackParams, defaultAction?: 'like' | 'dislike', onFeedback: (action: 'like' | 'dislike', comment: string) => Promise<void>, children: ReactNode }) {
  const [open, setOpen] = useState(false);
  const [action, setAction] = useState<'like' | 'dislike'>(initial?.feedback_type ?? defaultAction ?? 'like');
  const [comment, setComment] = useState(initial?.comment ?? '');
  const [running, setRunning] = useState(false);
  const [deleting, setDeleting] = useState(false);

  useEffect(() => {
    if (defaultAction && !initial) {
      setAction(defaultAction);
    }
  }, [defaultAction, initial]);

  useEffect(() => {
    if (initial) {
      setAction(initial.feedback_type);
      setComment(initial.comment);
    }
  }, [initial]);

  const disabled = running || deleting || !!initial;

  const container = usePortalContainer();

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      {children}
      <DialogContent container={container} className="space-y-4">
        <DialogHeader>
          <DialogTitle>
            Feedback
          </DialogTitle>
        </DialogHeader>
        <section className="space-y-2">
          <h6 className="text-sm font-bold">Do you like this answer</h6>
          <ToggleGroup disabled={disabled} className="w-max" type="single" value={action} onValueChange={value => setAction(value as any)}>
            <ToggleGroupItem value="like" className="data-[state=on]:text-success data-[state=on]:bg-success/10">
              <ThumbsUpIcon className="w-4 h-4 mr-2" />
              Like
            </ToggleGroupItem>
            <ToggleGroupItem value="dislike" className="data-[state=on]:text-destructive data-[state=on]:bg-destructive/10">
              <ThumbsDownIcon className="w-4 h-4 mr-2" />
              Dislike
            </ToggleGroupItem>
          </ToggleGroup>
        </section>
        <section>
          <Textarea
            placeholder="Comments..."
            value={comment}
            onChange={e => setComment(e.target.value)}
            disabled={disabled}
          />
        </section>
        <div className="flex w-full justify-end items-center gap-2">
          <Button
            className="gap-2"
            disabled={disabled}
            onClick={() => {
              setRunning(true);
              onFeedback(action, comment)
                .then(() => setOpen(false))
                .finally(() => {
                  setRunning(false);
                });
            }}>
            {running && <Loader2Icon className="w-4 h-4 animate-spin repeat-infinite" />}
            Add feedback
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
}
