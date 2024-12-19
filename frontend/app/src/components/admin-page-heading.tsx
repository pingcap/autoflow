'use client';

import { Breadcrumb, BreadcrumbItem, BreadcrumbLink, BreadcrumbList, BreadcrumbPage, BreadcrumbSeparator } from '@/components/ui/breadcrumb';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';
import { useSettingContext } from '@/components/website-setting-provider';
import { AlertCircleIcon, AlertTriangleIcon, CheckCircleIcon, HelpCircleIcon } from 'lucide-react';
import Link from 'next/link';
import { Fragment, type ReactNode } from 'react';

export interface BreadcrumbItem {
  title: ReactNode;
  url?: string;
  docsUrl?: string;
  alert?: {
    variant: 'success' | 'warning' | 'destructive';
    content: ReactNode;
  };
}

export interface TableHeadingProps {
  breadcrumbs?: BreadcrumbItem[];
}

export function AdminPageHeading ({ breadcrumbs }: TableHeadingProps) {
  const { title: siteTitle } = useSettingContext();
  return (
    <div className="mb-2 pl-8">
      {breadcrumbs && (
        <Breadcrumb>
          <BreadcrumbList>
            <BreadcrumbLink asChild>
              <Link href="/">
                {siteTitle}
              </Link>
            </BreadcrumbLink>
            {breadcrumbs.map((item, index) => (
              <Fragment key={index}>
                <BreadcrumbSeparator />
                <BreadcrumbItem>
                  {item.alert && <TooltipProvider>
                    <Tooltip>
                      <TooltipTrigger asChild>
                        {item.alert.variant === 'success'
                          ? <CheckCircleIcon className="text-success size-4" />
                          : item.alert.variant === 'warning'
                            ? <AlertTriangleIcon className="text-warning size-4" />
                            : item.alert.variant === 'destructive'
                              ? <AlertTriangleIcon className="text-destructive size-4" />
                              : <AlertCircleIcon className="text-muted-foreground size-4" />}
                      </TooltipTrigger>
                      <TooltipContent align='start'>
                        {item.alert.content}
                      </TooltipContent>
                    </Tooltip>
                  </TooltipProvider>}
                  {item.url
                    ? <BreadcrumbLink asChild><Link href={item.url}>{item.title}</Link></BreadcrumbLink>
                    : index === breadcrumbs.length - 1
                      ? <BreadcrumbPage>{item.title}</BreadcrumbPage>
                      : <span>{item.title}</span>}
                  {item.docsUrl
                    ? <a href={item.docsUrl} target="_blank"><HelpCircleIcon className="size-4" /></a>
                    : undefined}
                </BreadcrumbItem>
              </Fragment>
            ))}
          </BreadcrumbList>
        </Breadcrumb>
      )}
    </div>
  );
}
