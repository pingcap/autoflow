import type { Page, PageParams } from '@/lib/zod';

export async function listAllHelper<T> (api: (params: PageParams) => Promise<Page<T>>, idField: keyof T) {
  let page = 1;
  const chunks: Page<T>[] = [];

  while (true) {
    const current = await api({ page, size: 100 });
    chunks.push(current);
    if (page < current.pages) {
      page += 1;
    } else {
      break;
    }
  }

  const idSet = new Set();
  const result: T[] = [];

  for (const chunk of chunks) {
    for (const item of chunk.items) {
      if (!idSet.has(item[idField])) {
        idSet.add(item[idField]);
        result.push(item);
      }
    }
  }

  return result;
}