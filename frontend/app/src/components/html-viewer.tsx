'use client';

import Highlight from 'highlight.js/lib/core';
import html from 'highlight.js/lib/languages/xml';
import { useEffect, useState } from 'react';
import './code-theme.scss';

Highlight.registerLanguage('html', html);

export function HtmlViewer ({ value: propValue }: { value: string }) {
  const [value, setValue] = useState(() => propValue.replaceAll('<', '&lt;'));

  useEffect(() => {
    setValue(propValue);
    try {
      const { value: result } = Highlight.highlight(propValue, { language: 'html' });
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
