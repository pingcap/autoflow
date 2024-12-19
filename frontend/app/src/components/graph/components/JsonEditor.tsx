import { useActiveTheme } from '@/components/use-active-theme';
import * as monaco from 'monaco-editor';
import { forwardRef, useEffect, useImperativeHandle, useRef, useState } from 'react';

export const JsonEditor = forwardRef<monaco.editor.IStandaloneCodeEditor | undefined, { defaultValue: string, disabled?: boolean, id?: string }>(({ disabled, defaultValue, id }, forwardedRef) => {
  const [editor, setEditor] = useState<monaco.editor.IStandaloneCodeEditor | undefined>(undefined);

  useImperativeHandle(forwardedRef, () => editor, [editor]);

  const theme = useActiveTheme();

  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // SOLUTION: instead use dynamic imports on the client side
    if (ref) {
      const editor = monaco.editor.create(ref.current!, {
        value: '',
        language: 'json',
        automaticLayout: true,
        lineNumbers: 'off',
        tabSize: 2,
        theme: theme === 'dark' ? 'vs-dark' : undefined,
      });

      setEditor(editor);
      return () => editor.dispose();
    }
  }, []);

  useEffect(() => {
    if (editor) {
      editor.updateOptions({
        lineNumbers: disabled ? 'off' : 'on',
        lineDecorationsWidth: disabled ? 0 : undefined,
        readOnly: disabled,
      });
    }
  }, [editor, disabled]);

  useEffect(() => {
    if (editor) {
      if (defaultValue != null) {
        editor.setValue(defaultValue);
      }
    }
  }, [editor, defaultValue]);

  useEffect(() => {
    if (theme === 'dark') {
      editor?.updateOptions({
        theme: 'vs-dark',
      });
    } else {
      editor?.updateOptions({
        theme: 'vs',
      });
    }
  }, [theme]);

  return <div className="w-full h-full" id={id} ref={ref} style={{ height: '100%', width: '100%' }}></div>;
});

JsonEditor.displayName = 'JsonEditor';
