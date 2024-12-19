import { createContext, type Dispatch, type ReactNode, type SetStateAction, useContext, useEffect, useRef, useState } from 'react';
import { FormProvider, useFormContext } from 'react-hook-form';

type FieldsMap = Map<string, Set<string>>;
type FormSectionsContextValues = readonly [FieldsMap, Dispatch<SetStateAction<FieldsMap>>];
const FormSectionsContext = createContext<FormSectionsContextValues | undefined>(undefined);

const EMPTY_SET = new Set<string>();

export function FormSectionsProvider ({ children }: { children?: ReactNode }) {
  const context = useState<Map<string, Set<string>>>(() => new Map());
  return (
    <FormSectionsContext.Provider value={context}>
      {children}
    </FormSectionsContext.Provider>
  );
}

export function useFormSectionFields (section: string): ReadonlySet<string> {
  const [map] = useContext(FormSectionsContext) ?? [];
  return map?.get(section) ?? EMPTY_SET;
}

export function FormSection ({ value, children }: { value: string, children?: ReactNode }) {
  const { ...form } = useFormContext();
  const [_, setMap] = useContext(FormSectionsContext) ?? [];
  const deferred = useRef<(() => void)[]>([]);

  useEffect(() => {
    if (deferred.current.length > 0) {
      const current = [...deferred.current];
      deferred.current.splice(0, deferred.current.length);

      for (let fn of current) {
        fn();
      }
    }
  });

  return (
    <FormProvider
      {...form}
      control={{
        ...form.control,
        register: (name, options) => {
          deferred.current.push(() => {
            setMap?.(map => {
              if (map?.get(value)?.has(name)) {
                return map;
              }
              if (!map) {
                return new Map().set(value, new Set(name));
              } else {
                return new Map(map.set(value, new Set(map.get(value)).add(name)));
              }
            });
          });
          return form.control.register(name, options);
        },
      }}
    >
      {children}
    </FormProvider>
  );
}