import { GeneralSettingsFormContextProvider, type OnUpdateField } from '@/components/settings-form/context';
import { useLatestRef } from '@/components/use-latest-ref';
import { type ReactNode, useOptimistic, useTransition } from 'react';

export function GeneralSettingsForm<Data> ({ data, loading, readonly, onUpdate, children }: {
  data: Data,
  readonly: boolean,
  loading: boolean,
  onUpdate: (data: Readonly<Data>, path: (string | number | symbol)[]) => Promise<void>,
  children: ReactNode,
}) {
  const [updating, startTransition] = useTransition();
  const dataRef = useLatestRef(data);

  const [optimisticData, setOptimisticData] = useOptimistic(data);

  const onUpdateField: OnUpdateField<Data> = async (value, accessor) => {
    const data = accessor.set(dataRef.current, value);

    const updatePromise = onUpdate(data, accessor.path);
    startTransition(async () => {
      setOptimisticData(data);
      await updatePromise;
    });

    await updatePromise;
  };

  return (
    <GeneralSettingsFormContextProvider value={{ data: optimisticData, readonly, disabled: loading || updating, onUpdateField }}>
      {children}
    </GeneralSettingsFormContextProvider>
  );
}