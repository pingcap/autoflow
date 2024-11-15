import type { DeprecatedDatasource } from '@/api/datasources';
import { type ComponentType } from 'react';
import CreateFileDatasourceForm from './CreateFileDatasourceForm';
import CreateWebSinglePageDatasourceForm from './CreateWebSinglePageDatasourceForm';
import CreateWebSitemapDatasourceForm from './CreateWebSitemapDatasourceForm';

export interface CreateDatasourceFormProps {
  type: 'file' | 'web-single-page' | 'web-sitemap';
  onCreated?: (datasource: DeprecatedDatasource) => void;
  transitioning?: boolean;
  excludesLLM?: boolean;
}

export function CreateDatasourceForm ({ type, excludesLLM, onCreated, transitioning }: CreateDatasourceFormProps) {
  let FormComponent: ComponentType<{ onCreated?: (datasource: DeprecatedDatasource) => void, transitioning?: boolean, excludesLLM?: boolean }>;

  switch (type) {
    case 'file':
      FormComponent = CreateFileDatasourceForm;
      break;
    case 'web-sitemap':
      FormComponent = CreateWebSitemapDatasourceForm;
      break;
    case 'web-single-page':
      FormComponent = CreateWebSinglePageDatasourceForm;
      break;
  }

  return <FormComponent key={type} onCreated={onCreated} transitioning={transitioning} excludesLLM={excludesLLM} />;
}
