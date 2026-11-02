import { type APIRequestContext, expect, test } from '@playwright/test';

export async function loginViaApi ({ request }: { request: APIRequestContext }) {
  await test.step('Login via API', async () => {
    const usp = new URLSearchParams();
    usp.set('username', process.env.USERNAME);
    usp.set('password', process.env.PASSWORD);

    const response = await request.post('/api/v1/auth/login', {
      headers: { 'X-CSRF-Guard': '1' },
      data: usp.toString(),
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
      },
    });

    expect(response.ok()).toBe(true);
  });
}
