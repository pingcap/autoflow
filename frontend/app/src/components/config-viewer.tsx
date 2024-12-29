'use client';

import Highlight from 'highlight.js/lib/core';
import json from 'highlight.js/lib/languages/json';
import { useEffect, useState } from 'react';
import './code-theme.scss';

Highlight.registerLanguage('json', json);

export function ConfigViewer ({ value: propValue }: { value: any }) {
  const [value, setValue] = useState(() => {
    if (propValue === undefined) {
      return '';
    }
    try {
      return JSON.stringify(propValue, undefined, 2);
    } catch {
      return '/// FAILED TO STRINGIFY JSON';
    }
  });

  useEffect(() => {
    if (propValue === undefined) {
      return;
    }
    try {
      const string = JSON.stringify(propValue, undefined, 2);
      setValue(string);
      const { value: result } = Highlight.highlight(string, { language: 'json' });
      setValue(result);
    } catch {
    }
  }, [propValue]);

  return (
    <code className="block p-2 rounded bg-accent">
      <pre className="whitespace-pre-wrap text-xs font-mono" dangerouslySetInnerHTML={{ __html: value }} />
    </code>
  );
}