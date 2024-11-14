import { type Browser, expect, type Page, type Request, test } from '@playwright/test';

export const QUESTION = 'What is the content of sample.pdf?';

export function getChatRequestPromise (page: Page, baseURL: string) {
  return page.waitForRequest(request => request.url() === `${baseURL}/api/v1/chats` && request.method() === 'POST');
}

export async function testNewChat (browser: Browser, page: Page, chatRequest: Request, validatePageUrlAndTitle: boolean, feedbackLike?: boolean, testLangfuse: boolean = true) {
  await test.step('Wait page changes', async () => {
    if (validatePageUrlAndTitle) {
      await page.waitForURL(/\/c\/.+/);
      expect(await page.title()).toContain(QUESTION);
    }
    await page.getByRole('heading', { name: QUESTION }).waitFor({ state: 'visible' });
  });

  const streamText = await test.step('Wait for chat stop', async () => {
    const chatResponse = await chatRequest.response();
    expect(chatResponse.ok()).toBe(true);

    // Feedback button indicates chat ends.
    await page.getByRole('button', { name: 'Like This Answer', exact: true }).waitFor({ state: 'visible' });

    return await chatResponse.text();
  });

  await test.step('Check response text', async () => {
    const lastLine = streamText.split('\n').filter(t => !!t.trim()).slice(-1)[0];
    expect(lastLine).toMatch(/^2:/);
    const message = JSON.parse(lastLine.slice(2))[0].assistant_message;

    expect(message.finished_at).toBeTruthy();
    expect(message.content.trim().length).toBeGreaterThan(0);
  });

  if (typeof feedbackLike === 'boolean') {
    await test.step('Feedback', async () => {
      const feedbackButton = page.getByRole('button', { name: feedbackLike ? 'Like This Answer' : 'Dislike This Answer', exact: true });
      await feedbackButton.click();
      const dialog = page.getByRole('dialog', { name: 'Feedback' });

      await dialog.waitFor({ state: 'visible' });
      const comments = page.getByPlaceholder('Comments...');
      await comments.pressSequentially(feedbackLike ? 'Good Good Good' : 'Bad Bad Bad');
      await page.getByText('Add feedback', { exact: true }).click();

      await dialog.waitFor({ state: 'hidden' });
    });
  }

  if (testLangfuse) {
    await test.step('Validate langfuse URL', async () => {
      const debugInfoButton = await page.waitForSelector('button > svg.lucide-info');
      await debugInfoButton.click();

      const link = page.getByText('Langfuse Tracing');
      const href = await link.getAttribute('href');
      const langfusePage = await browser.newPage();
      await langfusePage.goto(href);
      await langfusePage.bringToFront();

      await langfusePage.getByLabel('Email').fill('langfuse@tidb.ai');
      await langfusePage.getByLabel('Password').fill('password');
      await langfusePage.getByTestId('submit-email-password-sign-in-form').click();

      await langfusePage.getByText('loading...').waitFor({ state: 'hidden' });

      await langfusePage.close();
    });
  }
}
