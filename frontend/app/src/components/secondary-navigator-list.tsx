'use client';

import { NextLink } from '@/components/nextjs/NextLink';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';
import * as TabsPrimitive from '@radix-ui/react-tabs';
import { usePathname } from 'next/navigation';
import { type ComponentProps, forwardRef, type ReactNode } from 'react';

const NAV_WIDTH = '14rem';

declare module 'react' {
  export interface CSSProperties {
    '--secondary-sidebar-width'?: string;
  }
}

export const SecondaryNavigatorLayout = forwardRef<HTMLDivElement, ComponentProps<'div'> & Pick<TabsPrimitive.TabsProps, 'defaultValue'>>(({ className, style, children, defaultValue, ...props }, ref) => {
  return (
    <TabsPrimitive.Tabs asChild orientation="vertical" defaultValue={defaultValue}>
      <div
        ref={ref}
        className={cn('flex gap-6 w-full', className)}
        style={{ '--secondary-sidebar-width': NAV_WIDTH, ...style }}
        {...props}
      >
        {children}
      </div>
    </TabsPrimitive.Tabs>
  );
});
SecondaryNavigatorLayout.displayName = 'SecondaryNavigatorLayout';

export const SecondaryNavigatorList = forwardRef<HTMLDivElement, ComponentProps<'div'>>(({ className, children, ...props }, ref) => {
  return (
    <TabsPrimitive.TabsList asChild loop>
      <div ref={ref} className={cn('space-y-2 w-[--secondary-sidebar-width] flex-shrink-0', className)} {...props}>
        {children}
      </div>
    </TabsPrimitive.TabsList>
  );
});
SecondaryNavigatorList.displayName = 'SecondaryNavigatorList';

export function SecondaryNavigatorLink ({ pathname, children }: { pathname: string, children: ReactNode }) {
  const current = usePathname();
  const active = current === pathname;

  return (
    <TabsPrimitive.Trigger value={pathname} asChild>
      <NextLink href={pathname} size="sm" className={cn('flex w-full justify-start', active ? 'font-semibold' : 'font-normal')} variant={active ? 'secondary' : 'ghost'}>
        {children}
      </NextLink>
    </TabsPrimitive.Trigger>
  );
}

export const SecondaryNavigatorItem = forwardRef<HTMLButtonElement, TabsPrimitive.TabsTriggerProps>(({ value, className, children, ...props }, ref) => {
  return (
    <TabsPrimitive.Trigger value={value} asChild>
      <Button
        ref={ref}
        {...props}
        variant="ghost"
        className={cn(
          'flex w-full justify-start',
          'font-normal data-[state=active]:font-semibold',
          /* data-[state=active]:secondary */'data-[state=active]:bg-secondary data-[state=active]:text-secondary-foreground data-[state=active]:hover:bg-secondary/80',
        )}
      >
        {children}
      </Button>
    </TabsPrimitive.Trigger>
  );
});

SecondaryNavigatorItem.displayName = 'SecondaryNavigatorTabsTrigger';

export const SecondaryNavigatorMain = forwardRef<HTMLDivElement, Omit<TabsPrimitive.TabsContentProps, 'value' | 'forceMount'> & { value?: string, strategy?: 'forceMount' | 'hidden' | 'mount' }>(({ value, strategy = 'mount', className, ...props }, ref) => {
  const classNames = cn('flex-1 overflow-x-hidden', className);
  if (value == null) {
    return <div ref={ref} className={classNames} {...props} />;
  } else {
    return (
      <TabsPrimitive.TabsContent
        ref={ref}
        value={value}
        forceMount={strategy !== 'mount' ? true : undefined}
        className={cn(classNames, strategy === 'hidden' && 'hidden data-[state=active]:block')}
        tabIndex={undefined}
        {...props}
      />
    );
  }
});

SecondaryNavigatorMain.displayName = 'SecondaryNavigatorContent';
