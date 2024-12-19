import { Loader } from '@/components/loader';
import { Maximize2Icon } from 'lucide-react';
import { editor } from 'monaco-editor';
import { forwardRef, lazy, type ReactNode, Suspense, useImperativeHandle, useMemo, useRef } from 'react';

const JsonEditor = lazy(() => import('./JsonEditor').then(res => ({ default: res.JsonEditor })));

export interface JsonFieldProps {
  label: ReactNode;
  defaultValue?: any;
  disabled?: boolean;
}

export interface JsonFieldInstance {
  value: any;
}

export const JsonField = forwardRef<JsonFieldInstance, JsonFieldProps>(({
  label,
  defaultValue,
  disabled,
}, ref) => {
  const editorRef = useRef<editor.IStandaloneCodeEditor | null | undefined>(undefined);
  const containerRef = useRef<HTMLDivElement>(null);

  const defaultValueString = useMemo(() => {
    return JSON.stringify(defaultValue, undefined, 2);
  }, [defaultValue]);

  const handleClickFullscreen = () => {
    containerRef.current?.requestFullscreen();
  };

  useImperativeHandle(ref, () => ({
    get value () {
      const value = editorRef.current?.getValue();
      if (value == null) {
        return undefined;
      }
      return JSON.parse(value);
    },
    set value (value: any) {
      editorRef.current?.setValue(JSON.stringify(value, undefined, 2));
    },
  }), []);

  return (
    <section>
      <h6 className="text-xs font-bold text-accent-foreground mb-1 flex items-center justify-between">
        {label}
        <button className="text-foreground/50 hover:text-foreground transition-colors" onClick={handleClickFullscreen}>
          <Maximize2Icon className="w-3 h-3" />
        </button>
      </h6>
      <div ref={containerRef} className="relative bg-card p-1 border rounded w-full h-80 overflow-auto focus-within:outline outline-2 -outline-offset-1 outline-primary">
        <Suspense fallback={<Loader loading>Initializing JSON editor...</Loader>}>
          <JsonEditor
            ref={editorRef}
            disabled={disabled}
            defaultValue={defaultValueString}
          />
        </Suspense>
      </div>
    </section>
  );
});

JsonField.displayName = 'JsonField';