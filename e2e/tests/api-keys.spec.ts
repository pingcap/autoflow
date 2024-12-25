import { expect, test } from '@playwright/test';
import { loginViaApi } from '../utils/login';

test.use({
  permissions: ['clipboard-read'],
});

test('API Keys Operations', async ({ page, baseURL }) => {

  await test.step('Login', async () => {
    await loginViaApi(page);
  });

  await test.step('Click Nav', async () => {
    await page.goto('/');
    await page.getByText('API Keys').click();
    await page.waitForURL('/api-keys');
  });

  const apiKey = await test.step('Create API Key', async () => {
    await page.getByText('Create', { exact: true }).click();

    const dialog = page.getByRole('dialog', { name: 'Create API Key' });
    await dialog.waitFor({ state: 'visible' });

    await page.fill('input[name=description]', 'Test API Key');
    await page.getByRole('button', { name: 'Create API Key', exact: true }).click();

    await dialog.waitFor({ state: 'hidden' });

    const apiKey = await page.evaluate('navigator.clipboard.readText()');
    expect(apiKey).not.toBeFalsy();

    return apiKey;
  });

  await test.step(`Try fetch with API Key ${apiKey}`, async () => {
    const response = await fetch(`${baseURL}/api/v1/api-keys`, {
      headers: {
        Authorization: `Bearer ${apiKey}`,
      },
    });

    if (!response.ok) {
      console.error(`${response.status} ${response.statusText}`, await response.text());
    }

    expect(response.ok).toBe(true);
  });

  await test.step('Delete API Key', async () => {
    const row = page.getByText('Test API Key').locator('..');
    await row.getByRole('button', { name: 'Delete' }).click();

    const dialog = page.getByRole('alertdialog', { name: 'Are you absolutely sure?' });
    await dialog.waitFor({ state: 'visible' });

    await dialog.getByRole('button', { name: 'Continue' }).click();
    await dialog.getByRole('button', { name: 'Continue' }).waitFor({ state: 'detached' });
    await dialog.waitFor({ state: 'hidden' });

    await row.waitFor({ state: 'detached' });
  });

  await test.step(`Try fetch with API Key (expect 401 Unauthorized)`, async () => {
    const response = await fetch(`${baseURL}/api/v1/api-keys`, {
      headers: {
        Authorization: `Bearer ${apiKey}`,
      },
    });
    expect(response.status).toBe(401);
  });
});
