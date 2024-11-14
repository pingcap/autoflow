import { expect, test } from '@playwright/test';
import { getChatRequestPromise, QUESTION, testNewChat } from '../utils/chat';
import { loginViaApi } from '../utils/login';

test.describe.serial('Chat', () => {
  test('From Home Page', async ({ browser, page, baseURL }) => {
    await test.step('Visit home page', async () => {
      await page.goto('/');
    });

    const chatRequest = await test.step('Input text and ask', async () => {
      await page.getByPlaceholder('Input your question here...').fill(QUESTION);

      // https://playwright.dev/docs/events#waiting-for-event
      const chatRequestPromise = getChatRequestPromise(page, baseURL);
      const trigger = page.locator('button', { has: page.locator('svg.lucide-arrow-up') });
      await trigger.click();

      await expect(trigger).toBeDisabled();
      return await chatRequestPromise;
    });

    await testNewChat(browser, page, chatRequest, true, true);
  });

  test('From Keyboard Shortcut', async ({ browser, page, baseURL }) => {
    await test.step('Visit home page', async () => {
      await page.goto('/');
    });

    const chatRequest = await test.step('Input text and ask', async () => {
      await page.keyboard.press('ControlOrMeta+k');
      await page.keyboard.insertText(QUESTION);

      // https://playwright.dev/docs/events#waiting-for-event
      const chatRequestPromise = getChatRequestPromise(page, baseURL);
      await page.keyboard.press('ControlOrMeta+Enter');
      return await chatRequestPromise;
    });

    await testNewChat(browser, page, chatRequest, true, false);
  });

  test('Admin Feedback Page', async ({ page }) => {
    await loginViaApi(page);
    await page.goto('/feedbacks');
    expect(await page.getByText('Good Good Good').count()).toBeGreaterThan(0);
    expect(await page.getByText('Bad Bad Bad').count()).toBeGreaterThan(0);
  });
});
