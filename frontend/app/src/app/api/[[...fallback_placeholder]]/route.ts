import { BASE_URL } from '@/lib/request';
import type { NextRequest } from 'next/server';

declare global {
  interface RequestInit {
    duplex?: 'half' | boolean;
  }
}

function handler (request: NextRequest) {
  const base = BASE_URL;

  if (!/^https?:\/\//.test(base)) {
    return Promise.reject(new Error(`BASE_URL must be a http(s) url to proxy requests.`));
  }

  const newUrl = new URL(originalUrl(request), base);

  const requestHeaders = new Headers(request.headers);
  requestHeaders.delete('Accept-Encoding');
  requestHeaders.delete('Host');
  requestHeaders.delete('X-Invoke-Output');
  requestHeaders.delete('X-Invoke-Path');
  requestHeaders.delete('X-Invoke-Query');
  requestHeaders.delete('X-Middleware-Invoke');

  return fetch(newUrl, {
    cache: 'no-cache',
    method: request.method,
    headers: requestHeaders,
    body: request.body,
    duplex: request.body instanceof ReadableStream ? 'half' : 'half',
  }).then(response => {
    console.log('[proxy]', request.method, newUrl.toString(), response.status, response.statusText, response.headers.get('Content-Type'));
    return response;
  }, error => {
    console.error('[proxy]', request.method, newUrl.toString(), error);
    return Promise.reject(error);
  });
}

function originalUrl (request: NextRequest) {
  const url = request.nextUrl;
  const usp = new URLSearchParams(url.searchParams);
  usp.delete('fallback_placeholder');
  const search = usp.toString();
  if (search) {
    return url.pathname + '?' + search;
  } else {
    return url.pathname;
  }
}

export { handler as GET, handler as POST, handler as DELETE, handler as HEAD, handler as PUT, handler as PATCH, handler as OPTIONS };

export const runtime = 'edge';

export const dynamic = 'force-dynamic';

export const maxDuration = 300;
