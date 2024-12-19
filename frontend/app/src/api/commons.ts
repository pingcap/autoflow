import { handleErrors, requestUrl } from '@/lib/request';

export async function setDefault (type: 'embedding-models' | 'llms' | 'reranker-models', id: number) {
  await fetch(requestUrl(`/api/v1/admin/${type}/${id}/set_default`), {
    method: 'PUT',
  }).then(handleErrors);
}
