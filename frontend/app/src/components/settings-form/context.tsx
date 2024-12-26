import type { GeneralSettingsFieldAccessor } from '@/components/settings-form/GeneralSettingsField';
import { createContext, type ReactNode, useContext } from 'react';

export type OnUpdateField<Data> = <T> (value: T, accessor: GeneralSettingsFieldAccessor<Data, T>) => void | Promise<void>;

export interface GeneralSettingsFormContextValues<Data> {
  data: Data;
  disabled: boolean;
  readonly: boolean;
  onUpdateField: OnUpdateField<Data>;
}

const GeneralSettingsFormContext = createContext<GeneralSettingsFormContextValues<any>>(null as any);

export function useGeneralSettingsFormContext<Data> () {
  return useContext(GeneralSettingsFormContext);
}

export function GeneralSettingsFormContextProvider<Data> ({ value, children }: { value: GeneralSettingsFormContextValues<Data>, children: ReactNode }) {
  return (
    <GeneralSettingsFormContext value={value}>
      {children}
    </GeneralSettingsFormContext>
  );
}
