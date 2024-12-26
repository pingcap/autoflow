import type { FormControlWidgetProps } from '@/components/form/control-widget';
import { cn } from '@/lib/utils';
import { Loader2Icon } from 'lucide-react';
import mergeRefs from 'merge-refs';
import type * as monaco from 'monaco-editor';
import { useTheme } from 'next-themes';
import { forwardRef, useEffect, useRef, useState } from 'react';

export interface CodeInputProps extends FormControlWidgetProps {
  className?: string;
  placeholder?: string;
  language: 'json' | 'markdown';
}

export const CodeInput = forwardRef<any, CodeInputProps>(({
  id,
  name,
  language,
  className,
  value,
  onChange,
  onBlur,
  disabled,
  placeholder,
  'aria-describedby': ariaDescribedBy,
  'aria-invalid': ariaInvalid,
}, forwardedRef) => {
  const monacoRef = useRef<typeof monaco>(undefined);
  const [editor, setEditor] = useState<monaco.editor.IStandaloneCodeEditor | undefined>(undefined);

  // useImperativeHandle(forwardedRef, () => editor, [editor]);

  const { theme } = useTheme();

  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (ref) {
      const ac = new AbortController();
      let editor: monaco.editor.IStandaloneCodeEditor | undefined;

      import('monaco-editor').then(monaco => {
        monacoRef.current = monaco;
        if (ac.signal.aborted) return;

        editor = monaco.editor.create(ref.current!, {
          value: value,
          language,
          automaticLayout: true,
          lineNumbers: 'off',
          glyphMargin: false,
          lineDecorationsWidth: 0,
          lineNumbersMinChars: 2,
          scrollBeyondLastLine: false,
          minimap: {
            enabled: false,
          },
          tabSize: 2,
          theme: theme === 'dark' ? 'vs-dark' : undefined,
        });

        setEditor(editor);
      });

      return () => {
        ac.abort();
        editor?.dispose();
      };
    }
  }, []);

  useEffect(() => {
    if (editor && onBlur) {
      const { dispose } = editor.onDidBlurEditorText(onBlur);
      return dispose;
    }
  }, [editor, onBlur]);

  useEffect(() => {
    if (editor && onChange) {
      const { dispose } = editor.onDidChangeModelContent(() => onChange(editor.getValue()));
      return dispose;
    }
  }, [editor, onChange]);

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
      const rValue = value || '';
      if (rValue !== editor.getValue()) {
        editor.setValue(rValue);
      }
    }
  }, [editor, value]);

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

  useEffect(() => {
    if (editor && placeholder && monacoRef.current) {
      const monaco = monacoRef.current;

      /**
       * Represents an placeholder renderer for monaco editor
       * Roughly based on https://github.com/microsoft/vscode/blob/main/src/vs/workbench/contrib/codeEditor/browser/untitledTextEditorHint/untitledTextEditorHint.ts
       */
      class PlaceholderContentWidget implements monaco.editor.IContentWidget {
        private static readonly ID = 'editor.widget.placeholderHint';

        private domNode: HTMLElement | undefined;

        constructor (
          private readonly placeholder: string,
          private readonly editor: monaco.editor.ICodeEditor,
        ) {
          // register a listener for editor code changes
          // ensure that on initial load the placeholder is shown
          this.onDidChangeModelContent();
        }

        private onDidChangeModelContent (): void {
          if (this.editor.getValue() === '') {
            this.editor.addContentWidget(this);
          } else {
            this.editor.removeContentWidget(this);
          }
        }

        getId (): string {
          return PlaceholderContentWidget.ID;
        }

        getDomNode (): HTMLElement {
          if (!this.domNode) {
            this.domNode = document.createElement('div');
            this.domNode.className = 'text-muted-foreground whitespace-pre-wrap text-italic opacity-70';
            this.domNode.style.width = 'max-content';
            this.domNode.style.pointerEvents = 'none';
            this.domNode.textContent = this.placeholder;
            this.editor.applyFontInfo(this.domNode);
          }

          return this.domNode;
        }

        getPosition (): monaco.editor.IContentWidgetPosition | null {
          return {
            position: { lineNumber: 1, column: 1 },
            preference: [monaco.editor.ContentWidgetPositionPreference.EXACT],
          };
        }
      }

      const widget = new PlaceholderContentWidget(placeholder, editor);
      const { dispose } = editor.onDidChangeModelContent(() => {
        if (editor.getValue() === '') {
          editor.addContentWidget(widget);
        } else {
          editor.removeContentWidget(widget);
        }
      });

      return () => {
        dispose();
      };
    }
  }, [editor, placeholder]);

  return (
    <>
      <div
        id={id}
        className={cn('border rounded w-full min-h-48', className, !editor && 'hidden')}
        ref={mergeRefs(ref, forwardedRef)}
        aria-describedby={ariaDescribedBy}
        aria-invalid={ariaInvalid}
      />
      {!editor && <div className="flex text-xs text-muted-foreground">Initializing code editor... <Loader2Icon className="size-4 animate-spin repeat-infinite" /></div>}
      <input className="hidden" value={value} readOnly name={name} />
    </>
  );
});
