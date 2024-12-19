import type { GeneralSettingsFieldAccessor } from '@/components/settings-form/GeneralSettingsField';
import type { KeyOfType } from '@/lib/typing-utils';
import { format, parse } from 'date-fns';

interface AccessorHelper<Row> {
  field<K extends string & keyof Row> (key: K): GeneralSettingsFieldAccessor<Row, Row[K]>;

  dateField<K extends string & KeyOfType<Row, Date>> (key: K): GeneralSettingsFieldAccessor<Row, string>;

  jsonTextField<K extends string & keyof Row> (key: K): GeneralSettingsFieldAccessor<Row, string>;

  nestedField<
    K0 extends string & KeyOfType<Row, Record<string, any>>,
    K1 extends string & keyof Row[K0]
  > (k0: K0, k1: K1): GeneralSettingsFieldAccessor<Row, Row[K0][K1]>;
}

export function createAccessorHelper<Row> (): AccessorHelper<Row> {
  return {
    field<K extends keyof Row> (key: K): GeneralSettingsFieldAccessor<Row, Row[K]> {
      return {
        path: [key],
        get (data) {
          return data[key];
        },
        set (data, value) {
          return {
            ...data,
            [key]: value,
          };
        },
      };
    },
    dateField<K extends string & KeyOfType<Row, Date>> (key: K): GeneralSettingsFieldAccessor<Row, string> {
      return {
        path: [key],
        get (data) {
          const date = data[key] as Date | undefined | null;
          if (!date) {
            return '--';
          }
          return format(date, 'yyyy-MM-dd HH:mm:ss');
        },
        set (data, value) {
          const date = parse(value, 'yyyy-MM-dd HH:mm:ss', new Date());
          return {
            ...data,
            [key]: date,
          };
        },
      };
    },
    jsonTextField<K extends string & KeyOfType<Row, Date>> (key: K): GeneralSettingsFieldAccessor<Row, string> {
      return {
        path: [key],
        get (data) {
          return JSON.stringify(data[key], undefined, 2);
        },
        set (data, value) {
          return {
            ...data,
            [key]: JSON.parse(value),
          };
        },
      };
    },
    nestedField<K0 extends string & KeyOfType<Row, Record<string, any>>, K1 extends string & keyof Row[K0]> (k0: K0, k1: K1): GeneralSettingsFieldAccessor<Row, Row[K0][K1]> {
      return {
        path: [k0, k1],
        get (row) {
          return row[k0]?.[k1] as any;
        },
        set (row, value) {
          return {
            ...row,
            [k0]: {
              ...row[k0],
              [k1]: value,
            },
          };
        },
      };
    },
  };
}