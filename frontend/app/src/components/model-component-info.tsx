import { cn } from '@/lib/utils';
import { Loader2Icon } from 'lucide-react';
import Link from 'next/link';

type ModelBase = {
  provider?: string
  id: number
  name: string
  model: string
}

export interface ModelComponentInfoProps<Model extends ModelBase> {
  className?: string;
  isLoading?: boolean;
  model: Model | null | undefined;
  url: (model: Model) => string;
  defaultName?: string;
}

export function ModelComponentInfo<Model extends ModelBase> ({ className, isLoading = false, model, url, defaultName }: ModelComponentInfoProps<Model>) {
  if (isLoading) {
    return <Loader2Icon className={cn('size-4 animate-spin repeat-infinite', className)} />;
  }

  if (!model) {
    return defaultName && <span className={cn('text-muted-foreground', className)}>{defaultName}</span>;
  }

  return (
    <span className={cn('flex gap-1 items-center', className)}>
      <Link className="font-bold underline" href={url(model)} target="_blank">{model.name}</Link>
      <span className="text-muted-foreground">
        <strong>{model.provider}</strong>:{model.model}
      </span>
    </span>
  );
}
