'use client';

import { type ButtonProps, buttonVariants } from '@/components/ui/button';
import { cn } from '@/lib/utils';
import Link, { type LinkProps } from 'next/link';
import { useRouter } from 'next/navigation';
import { forwardRef, MouseEvent, useTransition } from 'react';

export interface NextLinkProps extends Pick<LinkProps, 'prefetch' | 'scroll' | 'onClick' | 'replace'>, Pick<ButtonProps, 'className' | 'style' | 'variant' | 'size' | 'disabled' | 'children'> {
  disabled?: boolean;
  href: string;
}

export const NextLink = forwardRef<HTMLAnchorElement, NextLinkProps>(({ className, disabled: propDisabled, onClick, href, replace, scroll, variant, size, ...props }, ref) => {
  const [navigating, startTransition] = useTransition();
  const router = useRouter();

  const disabled = navigating || !!propDisabled;

  const handleClick = (event: MouseEvent<HTMLAnchorElement>) => {
    if (disabled) {
      event.preventDefault();
      return;
    }
    if (event.ctrlKey || event.shiftKey || event.metaKey || event.altKey) {
      event.persist();
      return;
    }
    onClick?.(event);
    if (event.defaultPrevented) {
      return;
    }
    event.preventDefault();

    startTransition(() => {
      if (replace) {
        router.replace(href, { scroll });
      } else {
        router.push(href, { scroll });
      }
    });
  };

  return (
    <Link
      {...props}
      onClick={handleClick}
      className={cn(buttonVariants({ variant, size }), 'aria-disabled:pointer-events-none aria-disabled:opacity-50', navigating && '!cursor-wait', className)}
      replace={replace}
      aria-disabled={disabled}
      href={href}
      scroll={scroll}
      ref={ref}
      role="button"
    />
  );
});

NextLink.displayName = 'NextLink';
