import type { FieldApi } from '@tanstack/react-form';
import { createContext, type Dispatch, type ReactNode, type SetStateAction, useContext, useEffect, useState } from 'react';

type FieldsMap = Map<string, Map<string, FieldApi<any, any>>>;
type FormSectionsContextValues = readonly [FieldsMap, Dispatch<SetStateAction<FieldsMap>>];
const FormSectionsContext = createContext<FormSectionsContextValues | undefined>(undefined);

const EMPTY_SET = new Map<string, FieldApi<any, any>>();

export function FormSectionsProvider ({ children }: { children?: ReactNode }) {
  const context = useState<Map<string, Map<string, FieldApi<any, any>>>>(() => new Map());
  return (
    <FormSectionsContext value={context}>
      {children}
    </FormSectionsContext>
  );
}

export function useFormSectionFields (section: string): ReadonlyMap<string, FieldApi<any, any>> {
  const [map] = useContext(FormSectionsContext) ?? [];
  return map?.get(section) ?? EMPTY_SET;
}

interface FormSectionContextValues {
  register (field: FieldApi<any, any>): () => void;
}

const FormSectionContext = createContext<FormSectionContextValues>({
  register (field: FieldApi<any, any>): () => void {
    return () => {};
  },
});

export function FormSection ({ value, children }: { value: string, children?: ReactNode }) {
  const [_, setMap] = useContext(FormSectionsContext) ?? [];

  const register = (field: FieldApi<any, any>) => {
    setMap?.(map => {
      map = new Map(map);
      const fieldMap = new Map(map.get(value));
      map.set(value, fieldMap);

      fieldMap.set(field.name, field);

      return map;
    });

    return () => {
      setMap?.(map => {
        if (!map.get(value)?.has(field.name)) {
          return map;
        }
        const fieldMap = new Map(map.get(value));
        map.set(value, fieldMap);

        fieldMap.delete(field.name);

        return map;
      });
    };
  };

  return (
    <FormSectionContext value={{ register }}>
      {children}
    </FormSectionContext>
  );
}

export function useRegisterFieldInFormSection (field: FieldApi<any, any, any, any>) {
  const { register } = useContext(FormSectionContext);
  useEffect(() => {
    return register(field);
  }, [field]);
}
