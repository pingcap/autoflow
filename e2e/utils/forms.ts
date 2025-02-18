import { expect, type Page, test } from '@playwright/test';

export async function selectOption (page: Page, name: string, value: string | RegExp, clickWindow = false) {
  await test.step(`Select field ${name}`, async () => {
    await page.getByRole('button', { name: name, exact: true }).click();
    await page.getByRole('option', { name: value }).click();
    if (clickWindow) {
      await page.click('body');
    }
    await expect(page.getByRole('button', { name: name, exact: true })).toHaveText(value);
  });
}

export async function turnSwitch (page: Page, name: string, on: boolean = true) {
  await test.step(`Turn ${on ? 'on' : 'off'} switch ${name}`, async () => {
    const locator = page.getByRole('switch', { name: name, exact: true });
    if (on) {
      if (await locator.getAttribute('aria-checked') === 'true') {
        return;
      }
      await locator.click();
      await expect(locator).toHaveAttribute('aria-checked', 'true');
    } else {
      if (await locator.getAttribute('aria-checked') === 'false') {
        return;
      }
      await locator.click();
      await expect(locator).toHaveAttribute('aria-checked', 'false');
    }
  });
}

export async function checkCheckbox (page: Page, name: string, on: boolean = true) {
  await test.step(`${on ? 'Check' : 'Uncheck'} checkbox ${name}`, async () => {
    const locator = page.getByRole('checkbox', { name: name, exact: true });
    if (on) {
      await locator.check();
    } else {
      await locator.uncheck();
    }
  });
}
