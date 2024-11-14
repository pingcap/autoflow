import { type Browser, expect, type Locator, type Page, test } from '@playwright/test';
import { getChatRequestPromise, QUESTION, testNewChat } from '../utils/chat';

test('JS Widget', async ({ browser, page }) => {
  await page.goto('/');
  await page.getByRole('button', { name: 'Ask AI' }).waitFor({ state: 'visible' });
  expect(await page.evaluate('tidbai')).toMatchObject({ open: false });
});

test('Embedded JS Widget with trigger button', async ({ browser, page }) => {
  const trigger = await test.step('Wait trigger visible and tidbai object ready', async () => {
    await page.goto('http://localhost:4001/widget.html');
    const trigger = page.getByRole('button', { name: 'Ask AI' });
    await trigger.waitFor({ state: 'visible' });
    expect(await page.evaluate('tidbai')).toMatchObject({ open: false });
    return trigger;
  });

  const dialog = await test.step('Click and show dialog', async () => {
    await trigger.click();

    const dialog = page.getByRole('dialog', { name: 'Ask AI' });
    await dialog.waitFor({ state: 'visible' });

    return dialog;
  });

  await testWidgetChat(browser, page, dialog);
});

// Used by docs.pingcap.com
test('Embedded JS Widget controlled by js', async ({ browser, page }) => {
  await test.step('Wait trigger visible and tidbai object ready', async () => {
    await page.goto('http://localhost:4001/widget-controlled.html');
    const trigger = page.getByRole('button', { name: 'Ask AI' });
    await trigger.waitFor({ state: 'detached' });
    expect(await page.evaluate('window.tidbai')).toMatchObject({ open: false });
  });

  const dialog = await test.step('JS api call and show dialog', async () => {
    await page.evaluate('tidbai.open = true');

    const dialog = page.getByRole('dialog', { name: 'Ask AI' });
    await dialog.waitFor({ state: 'visible' });

    return dialog;
  });

  await testWidgetChat(browser, page, dialog);
});

async function testWidgetChat (browser: Browser, page: Page, dialog: Locator) {
  await test.step('Fill in question', async () => {
    const input = dialog.getByPlaceholder('Input your question here...');
    await input.focus();
    await input.fill(QUESTION);
  });

  const chatRequestPromise = await test.step('Trigger ask by press ControlOrMeta+Enter', async () => {
    const chatRequestPromise = getChatRequestPromise(page, 'http://127.0.0.1:3000');
    await page.keyboard.press('ControlOrMeta+Enter');

    return chatRequestPromise;
  });

  await testNewChat(browser, page, chatRequestPromise, false, false);
}