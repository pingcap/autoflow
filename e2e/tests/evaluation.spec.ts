import { expect, type Page, test } from '@playwright/test';
import { loginViaApi } from '../utils/login';

test.describe('Evaluation Dataset Management', () => {
  test('Create dataset with CSV', async ({ page }) => {
    await loginViaApi(page);

    await createEvaluationDataset(page, 'Example Dataset from CSV', 'res/sample-evaluation-dataset.csv');

    // Data from CSV
    await expect(page.getByText('Example Query')).toBeVisible();
  });

  test('Create dataset from scratch', async ({ page }) => {
    await loginViaApi(page);
    await createEvaluationDataset(page, 'Example Dataset from scratch', 'res/sample-evaluation-dataset.csv');

    // Empty dataset items list.
    await expect(page.getByText('Empty List')).toBeVisible();
  });

  test('Delete dataset', async ({ page }) => {
    await loginViaApi(page);

    await createEvaluationDataset(page, 'Example Dataset to delete');

    await page.goto('/evaluation/datasets');

    await expect(page.getByRole('row').filter({ hasText: 'Example Dataset to delete' })).toBeVisible();

    await page.getByRole('row').filter({ hasText: 'Example Dataset to delete' }).locator('button').last().click();
    await page.getByRole('menuitem', { name: 'Delete' }).click();
    await page.getByRole('button', { name: 'Continue' }).click();

    await page.getByRole('row').filter({ hasText: 'Example Dataset to delete' }).waitFor({ state: 'detached' });
  });

  test('Mutate dataset items', async ({ page }) => {
    await loginViaApi(page);
    const datasetId = await createEvaluationDataset(page, 'Example Dataset to update');

    // Empty dataset items list.
    await expect(page.getByText('Empty List')).toBeVisible();

    await test.step('Add item', async () => {
      await page.getByRole('button', { name: 'New Item' }).click();
      await page.waitForURL(`/evaluation/datasets/${datasetId}/items/new`);
      await page.getByRole('textbox', { name: 'Query' }).fill('Example Query');
      await page.getByRole('textbox', { name: 'Reference' }).fill('Example Reference');
      await page.getByRole('button', { name: 'Create' }).click();
      await page.waitForURL(`/evaluation/datasets/${datasetId}`);

      await page.getByRole('row').filter({ hasText: 'Example Query' }).waitFor({ state: 'attached' });
      await page.getByRole('row').filter({ hasText: 'Example Reference' }).waitFor({ state: 'attached' });
    });

    await test.step('Delete Item', async () => {
      await page.getByRole('row').filter({ hasText: 'Example Query' }).locator('button').last().click();
      await page.getByRole('menuitem', { name: 'Delete' }).click();
      await page.getByRole('button', { name: 'Continue' }).click();

      await page.getByRole('row').filter({ hasText: 'Example Query' }).waitFor({ state: 'detached' });
      await page.getByRole('row').filter({ hasText: 'Example Reference' }).waitFor({ state: 'detached' });
    });
  });

});

async function createEvaluationDataset (page: Page, name: string, file?: string) {
  await page.goto('/');

  await test.step('Navigate to Create Evaluation Dataset Page', async () => {
    await page.getByRole('button', { name: 'Evaluation' }).click();
    await page.getByRole('link', { name: 'Datasets' }).click();
    await page.waitForURL('/evaluation/datasets');
    await page.getByRole('button', { name: 'New Evaluation Dataset' }).click();
    await page.waitForURL('/evaluation/datasets/create');
  });

  return await test.step('Fill in form and submit', async () => {
    await page.getByRole('textbox', { name: 'Name' }).fill(name);
    if (file) {
      await page.locator('[name=upload_file]').setInputFiles(file);
    }
    await page.getByRole('button', { name: 'Create' }).click();
    await page.waitForURL(/\/evaluation\/datasets\/\d+/);

    const [_, idString] = /\/evaluation\/datasets\/(\d+)/.exec(page.url());

    return parseInt(idString);
  });
}
