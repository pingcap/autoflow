import { type DatasourceType, isDatasourceType } from '@/components/datasource/types';
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs';
import type { ReactNode } from 'react';

export interface DatasourceTypeTabsProps {
  className?: string;
  type?: DatasourceType;
  onTypeChange?: (value: DatasourceType) => void;
  children?: ReactNode;
}

export function DatasourceTypeTabs ({ className, type, onTypeChange, children }: DatasourceTypeTabsProps) {
  return (
    <Tabs className={className} value={type} onValueChange={value => {
      isDatasourceType(value) && onTypeChange?.(value);
    }}>
      <TabsList>
        <TabsTrigger value="file">
          File
        </TabsTrigger>
        <TabsTrigger value="web-single-page">
          Web Single Page
        </TabsTrigger>
        <TabsTrigger value="web-sitemap">
          Web Sitemap
        </TabsTrigger>
      </TabsList>
      {children}
    </Tabs>
  );
}
