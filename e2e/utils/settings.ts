import type { Page } from '@playwright/test';

export async function submitAndWaitSavedByLabel (page: Page, label: string) {
  const button = page.getByText(label, { exact: true }).locator('..').locator('..').getByRole('button', { name: 'Save', exact: true });

  // Click the save button in the field form
  await button.click();

  // Wait the save button to be vanished. (Saved)
  await button.waitFor({ state: 'hidden' });
}