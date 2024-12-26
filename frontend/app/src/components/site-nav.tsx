'use client';

import { AlertDialog, AlertDialogAction, AlertDialogCancel, AlertDialogContent, AlertDialogDescription, AlertDialogFooter, AlertDialogHeader, AlertDialogTitle, AlertDialogTrigger } from '@/components/ui/alert-dialog';
import { Button, type ButtonProps } from '@/components/ui/button';
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible';
import { SidebarGroup, SidebarGroupContent, SidebarGroupLabel, SidebarMenu, SidebarMenuBadge, SidebarMenuButton, SidebarMenuItem, SidebarMenuSub, SidebarMenuSubButton, SidebarMenuSubItem } from '@/components/ui/sidebar';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';
import { ChevronDownIcon, TrashIcon } from 'lucide-react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';

import { type ComponentType, Fragment, type ReactElement, type ReactNode } from 'react';

export interface NavGroup {
  title?: ReactNode;
  items: NavItem[];
  sectionProps?: {
    className?: string;
  };
}

export interface NavBaseItem {
  icon?: ComponentType<{ className?: string }>;
  title: ReactNode;
  details?: ReactNode;
  className?: string;
  disabled?: ReactNode | boolean;
}

export interface NavLinkItem extends NavBaseItem {
  parent?: undefined;
  custom?: undefined;
  href: string;
  exact?: boolean;
  variant?: ButtonProps['variant'] | ((active: boolean) => ButtonProps['variant']);
  onDelete?: () => void;
  deleteResourceName?: string;
}

export interface NavParentItem extends NavBaseItem {
  parent: true;
  key: string;
  custom?: undefined;
  variant?: ButtonProps['variant'] | ((active: boolean) => ButtonProps['variant']);
  children: (NavLinkItem | CustomItem)[];
}

export interface CustomItem {
  parent?: undefined;
  custom: true;
  key: string;
  children: ReactNode;
}

export type NavItem = NavLinkItem | NavParentItem | CustomItem;

const isCustomItem = (item: NavItem): item is CustomItem => !!item.custom;
const isParentItem = (item: NavItem): item is NavParentItem => !!item.parent;
const isNavLinkItem = (item: NavItem): item is NavLinkItem => 'href' in item && !('children' in item);

export interface SiteNavProps {
  groups: NavGroup[];
}

export function SiteNav ({ groups }: SiteNavProps) {
  const pathname = usePathname() ?? '';

  return (
    <TooltipProvider>
      {groups.map((group, index) => (
        <Fragment key={index}>
          <SiteNavGroup group={group} current={pathname} />
        </Fragment>
      ))}
    </TooltipProvider>
  );
}

function SiteNavGroup ({ group, current }: { group: NavGroup, current: string }) {
  const { sectionProps: { className: sectionClassName, ...restSectionProps } = {} } = group;
  return (
    <SidebarGroup>
      {group.title && <SidebarGroupLabel>{group.title}</SidebarGroupLabel>}
      <SidebarGroupContent>
        <SidebarMenu>
          {renderItems(group.items, current)}
        </SidebarMenu>
      </SidebarGroupContent>
    </SidebarGroup>
  );
}

function isActive (current: string, item: NavLinkItem) {
  return current === item.href || (!item.exact && current.startsWith(item.href));
}

const renderItems = (items: NavItem[], current: string) => {
  return (
    <>
      {items.map(item => (
        isCustomItem(item)
          ? <Fragment key={item.key}>{item.children}</Fragment>
          : isParentItem(item)
            ? <SiteParentItem key={item.key} current={current} active={!!item.children.find(child => isNavLinkItem(child) && isActive(current, child))} item={item} />
            : <SiteNavLinkItem key={item.href} item={item} active={isActive(current, item)} />
      ))}
    </>
  );
};

const renderParentBaseItemContent = (item: NavParentItem) => {
  return (
    <>
      {item.icon && <item.icon className="opacity-70" />}
      {item.title}
    </>
  );
};

const renderParentItemChildren = (current: string, item: NavParentItem) => {
  return (
    <>
      {item.children.map(item => (
        <Fragment key={isCustomItem(item) ? item.key : item.href}>
          {isCustomItem(item)
            ? <Fragment key={item.key}>{item.children}</Fragment>
            : <SiteNavLinkItem key={item.href} item={item} active={isActive(current, item)} sub />}
        </Fragment>
      ))}
    </>
  );
};

const renderBaseItemContent = (item: NavBaseItem) => {
  return (
    <>
      {item.icon && <item.icon className="opacity-70" />}
      {item.title}
    </>
  );
};

function SiteParentItem ({ current, item, active }: { current: string, item: NavParentItem, active: boolean }) {
  let el: ReactElement<any> = renderParentBaseItemContent(item);

  if (item.disabled && typeof item.disabled !== 'boolean') {
    el = (
      <Tooltip>
        <TooltipTrigger asChild disabled={!!item.disabled}>
          {el}
        </TooltipTrigger>
        <TooltipContent>
          {item.disabled}
        </TooltipContent>
      </Tooltip>
    );
  }

  return (
    <Collapsible defaultOpen={active} className="group" asChild>
      <SidebarMenuItem>
        <CollapsibleTrigger asChild disabled={!!item.disabled}>
          <SidebarMenuButton isActive={active}>
            {el}
          </SidebarMenuButton>
        </CollapsibleTrigger>
        <SidebarMenuBadge>
          <ChevronDownIcon className="size-4 transition-transform group-data-[state=open]:rotate-180" />
        </SidebarMenuBadge>
        <CollapsibleContent asChild>
          <SidebarMenuSub>
            {renderParentItemChildren(current, item)}
          </SidebarMenuSub>
        </CollapsibleContent>
      </SidebarMenuItem>
    </Collapsible>
  );
}

function SiteNavLinkItem ({ item, active, sub = false }: { item: NavLinkItem, active: boolean, sub?: boolean }) {
  let el: ReactElement<any>;
  let badge: ReactNode | undefined;

  if (!!item.disabled) {
    el = renderBaseItemContent(item);
  } else {
    el = renderBaseItemContent(item);
  }

  if (item.details) {
    badge = item.details;
  }

  if (item.onDelete) {
    el = (
      <div className="flex gap-2 items-center">
        <div className="flex-1 overflow-hidden text-ellipsis text-nowrap">
          {el}
        </div>
        <AlertDialog>
          <AlertDialogTrigger asChild>
            <Button className="flex-shrink-0 w-max h-max rounded-full p-1 hover:bg-transparent" size="icon" variant="ghost" disabled={!!item.disabled}>
              <TrashIcon className="w-3 h-3 opacity-20 hover:opacity-60" />
            </Button>
          </AlertDialogTrigger>
          <AlertDialogContent>
            <AlertDialogHeader>
              <AlertDialogTitle>Are you sure to delete {item.deleteResourceName}?</AlertDialogTitle>
              <AlertDialogDescription>
                This action cannot be undone.
              </AlertDialogDescription>
            </AlertDialogHeader>
            <AlertDialogFooter>
              <AlertDialogCancel>Cancel</AlertDialogCancel>
              <AlertDialogAction onClick={item.onDelete}>Delete</AlertDialogAction>
            </AlertDialogFooter>
          </AlertDialogContent>
        </AlertDialog>
      </div>
    );
  }

  const MenuItem = sub ? SidebarMenuSubItem : SidebarMenuItem;
  const MenuButton = sub ? SidebarMenuSubButton : SidebarMenuButton;

  el = (
    <MenuButton asChild isActive={active} disabled={!!item.disabled}>
      <Link href={item.href}>
        {el}
      </Link>
    </MenuButton>
  );

  if (item.disabled && typeof item.disabled !== 'boolean') {
    el = (
      <Tooltip>
        <TooltipTrigger asChild disabled={!!item.disabled}>
          {el}
        </TooltipTrigger>
        <TooltipContent>
          {item.disabled}
        </TooltipContent>
      </Tooltip>
    );
  }

  return (
    <MenuItem>
      {el}
      {badge && <SidebarMenuBadge className="pointer-events-auto">{badge}</SidebarMenuBadge>}
    </MenuItem>
  );
}
