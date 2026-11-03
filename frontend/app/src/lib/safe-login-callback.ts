/** Resolve the browser's untrusted Referer to a local post-login route. */
export function safeLoginCallback (url: string | undefined, origin: string): string {
  if (!url || url.length > 2048 || /[\\\u0000-\u0020\u007f]/.test(url)) return '/';
  try {
    const target = new URL(url, origin);
    const decodedPath = decodeURIComponent(target.pathname);
    if (target.origin !== new URL(origin).origin || !/^https?:$/.test(target.protocol)
      || target.username || target.password || decodedPath.startsWith('//')
      || decodedPath.includes('\\') || /^\/auth(?:\/|$)/.test(decodedPath)) return '/';
    return target.pathname + target.search + target.hash;
  } catch {
    return '/';
  }
}
