import { expect, test } from '@playwright/test';
import type { APIResponse } from 'playwright-core';
import { loginViaApi } from '../utils/login';

// let key: string = '';
//
// test.beforeAll(async ({ request }) => {
//   const response = await request.post('/api/v1/api-leys', {
//     data: JSON.stringify({
//       description: 'E2E API Tests',
//     }),
//   });
//
//   const { api_key } = await response.json();
//   key = api_key;
//
//   console.log(`Created API Key, ${key}`);
// });

test.describe('API', () => {
  test('Bootstrap Status API', async ({ request }) => {
    await expectOk(request.get('/api/v1/system/bootstrap-status'));
  });

  test('Retrieve Entity or Entites', async ({ request }) => {
    async function expectGetOkStep (url: string) {
      await test.step(url, async () => {
        const response = await request.get(url);

        if (!response.ok()) {
          console.error(`${response.status()} ${response.statusText()}`, await response.text());
        }

        await expectOk(response);
      });
    }

    await loginViaApi({ request });

    //
    // =====
    // User

    await expectGetOkStep('/api/v1/chats');

    await expectGetOkStep('/api/v1/site-config');

    await expectGetOkStep('/api/v1/api-keys');

    //
    // =====
    // Admin

    await expectGetOkStep('/api/v1/admin/chat-engines');

    // Deprecated?
    await expectGetOkStep('/api/v1/admin/documents');
    await expectGetOkStep('/api/v1/admin/documents?page=1&size=10&language=en');
    await expectGetOkStep('/api/v1/admin/documents?page=1&size=10&source_uri=system-variables&created_at_start=2024-07-04T11:58:00&created_at_end=2024-07-04T12:58:00&updated_at_start=2024-08-22T14:36:59&updated_at_end=2024-08-23T14:37:59&last_modified_at_start=2024-07-04T11:57:58&last_modified_at_end=2024-07-05T11:57:59&name=System&mime_type=text/markdown&index_status=failed');

    await expectGetOkStep('/api/v1/admin/knowledge_bases');

    await expectGetOkStep('/api/v1/admin/knowledge_bases/1/documents');
    await expectGetOkStep('/api/v1/admin/knowledge_bases/1/documents?page=1&size=10&language=en');
    await expectGetOkStep('/api/v1/admin/knowledge_bases/1/documents?page=1&size=10&source_uri=system-variables&created_at_start=2024-07-04T11:58:00&created_at_end=2024-07-04T12:58:00&updated_at_start=2024-08-22T14:36:59&updated_at_end=2024-08-23T14:37:59&last_modified_at_start=2024-07-04T11:57:58&last_modified_at_end=2024-07-05T11:57:59&name=System&mime_type=text/markdown&index_status=failed');

    await expectGetOkStep('/api/v1/admin/site-settings');

    await expectGetOkStep('/api/v1/admin/feedbacks');

    await expectGetOkStep('/api/v1/admin/llms');
    await expectGetOkStep('/api/v1/admin/llms/options');

    await expectGetOkStep('/api/v1/admin/embedding-models');
    await expectGetOkStep('/api/v1/admin/embedding-models/options');

    await expectGetOkStep('/api/v1/admin/reranker-models');
    await expectGetOkStep('/api/v1/admin/reranker-models/options');

    await expectGetOkStep('/api/v1/admin/retrieve/documents?chat_engine=1&question=what%20is%20tidb&chat_engine=default&top_k=5');
    await expectGetOkStep('/api/v1/admin/embedding_retrieve?chat_engine=1&question=what%20is%20tidb&chat_engine=default&top_k=5');

    await expectGetOkStep('/api/v1/admin/evaluation/datasets');
    await expectGetOkStep('/api/v1/admin/evaluation/tasks');
  });
});

async function expectOk (response: APIResponse | Promise<APIResponse>) {
  expect((await response).ok()).toBe(true);
}
