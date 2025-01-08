'use client';

import Highlight from 'highlight.js/lib/core';
import python from 'highlight.js/lib/languages/python';
import { useEffect, useState } from 'react';
import './code-theme.scss';

Highlight.registerLanguage('python', python);

export function PythonViewer ({ value: propValue }: { value: string }) {
  const [value, setValue] = useState(() => propValue.replaceAll('<', '&lt;'));

  useEffect(() => {
    setValue(propValue);
    try {
      const { value: result } = Highlight.highlight(propValue, { language: 'python' });
      setValue(result);
    } catch {
    }
  }, [propValue]);

  return (
    <code>
      <pre className="whitespace-pre-wrap text-xs font-mono" dangerouslySetInnerHTML={{ __html: value }} />
    </code>
  );
}
