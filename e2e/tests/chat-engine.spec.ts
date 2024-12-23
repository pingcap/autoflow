import { expect, type Page, test } from '@playwright/test';
import { loginViaApi } from '../utils/login';

test.describe('Chat Engine', () => {
  test.describe('Configurations', () => {
    test('Create with default configuration', async ({ page }) => {
      await loginViaApi(page);
      await page.goto('/chat-engines');
      await page.getByRole('button', { name: 'New Chat Engine' }).click();
      await page.waitForURL('/chat-engines/new');

      const name = 'All default configuration';

      // Fill in name
      await page.getByRole('textbox', { name: 'Name' }).fill(name);

      // Goto retrieval tab
      await page.getByRole('tab', { name: 'Retrieval' }).click();

      // Select default knowledge base
      await page.getByRole('button', { name: 'Select Knowledge Base' }).click();
      await page.getByRole('option', { name: 'My Knowledge Base' }).click();
      await expect(page.getByRole('button', { name: 'Select Knowledge Base' })).toHaveText(/My Knowledge Base/);

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

// TODO: The selectors are tricky. Update the select component to simplify the validation.
async function checkChatEngineAvailability (page: Page, id: number, name: string) {
  await page.goto('/');

  // Select the 'Select Chat Engine' combobox
  const selector = page.getByRole('combobox').and(page.getByText('Select Chat Engine', { exact: true }).locator('..'));
  await selector.click();
  await page.getByRole('option', { name: name }).click();

  // Input question
  await page.getByPlaceholder('Input your question here...').fill('Hello');

  // Send message
  await page.keyboard.press('ControlOrMeta+Enter');

  // Wait page url to be changed. When changed, the chat was created correctly.
  // Ignore the returned message which is not important.
  await page.waitForURL(/\/c\/.+$/);
}
