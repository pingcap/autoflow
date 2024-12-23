import { expect, type Page, test } from '@playwright/test';
import { loginViaApi } from '../utils/login';

test.describe('Chat Engine', () => {
  test.describe('Configurations', () => {
    test('Create with default configuration', async ({ page }) => {
      await loginViaApi(page);
      await page.goto('/chat-engines');
      await page.getByRole('button', { name: 'Create Chat Engine' }).click();
      await page.waitForURL('/chat-engines/new');

      const name = 'All default configuration';

      // Fill in name
      await page.getByRole('textbox', { name: 'Name' }).fill(name);

      // Goto retrieval tab
      await page.getByRole('tab', { name: 'Retrieval' }).click();

      // Select default knowledge base
      await page.getByRole('button', { name: 'Select Knowledge Base' }).click();
      await page.getByRole('option', { name: 'Default' }).click();

      // Create
      await page.getByRole('button', { name: 'Create Chat Engine' }).click();
      await page.waitForURL(/\/chat-engines\/\d+$/);

      // Wait for created
      const [_, idString] = /\/chat-engines\/(\d+)$/.exec(page.url());
      const chatEngineId = parseInt(idString);

      await checkChatEngineAvailability(page, chatEngineId, name);
    });
  });
});

async function checkChatEngineAvailability (page: Page, id: number, name: string) {
  await page.goto('/');

  await page.getByRole('combobox', { name: 'Select Chat Engine' }).click();
  await page.getByRole('option', { name: name }).click();

  await expect(page.getByRole('combobox', { name: 'Select Chat Engine' })).toHaveValue(name);

  await page.getByPlaceholder('Input your question here...').fill('Hello');
  await page.keyboard.press('ControlOrMeta+Enter');

  await page.waitForURL(/\/c\/.+$/);
}
