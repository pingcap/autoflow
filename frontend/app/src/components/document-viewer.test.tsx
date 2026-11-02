/** @jest-environment node */
import { jest } from '@jest/globals';
import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import Highlight from 'highlight.js/lib/core';
import { DocumentViewer } from './document-viewer';

const payload = '<img src=x onerror=alert(1)><script>alert(2)</script>';

afterEach(() => jest.restoreAllMocks());

test('the initial render never inserts raw Markdown as HTML', () => {
  const html = renderToStaticMarkup(createElement(DocumentViewer, { content: payload, mime: 'text/markdown' }));
  expect(html).not.toContain('<img');
  expect(html).not.toContain('<script');
  expect(html).toContain('&lt;');
});

test('a failed highlighter falls back to React-escaped text', () => {
  jest.spyOn(Highlight, 'highlight').mockImplementation(() => { throw new Error('highlight failed'); });
  const html = renderToStaticMarkup(createElement(DocumentViewer, { content: payload, mime: 'text/markdown' }));
  expect(html).not.toContain('<img');
  expect(html).not.toContain('<script');
  expect(html).toContain('&lt;img');
});
