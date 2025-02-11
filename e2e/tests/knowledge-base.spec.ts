import { expect, type Page, test } from '@playwright/test';
import { selectOption, turnSwitch } from '../utils/forms';
import { loginViaApi } from '../utils/login';

test.describe('Knowledge Base', () => {
  test('Configure Data Sources', async ({ page }) => {
    const kbId = await createFeaturedKnowledgeBase(page, 'KnowledgeBase 1', true);

    await test.step('Configure Files Data Source', async () => {
      await page.getByRole('button', { name: 'Upload Files' }).click();
      await page.waitForURL(/data-sources\/new\?type=file/);

      await page.setInputFiles('[name=files]', 'res/sample.pdf');

      await page.getByRole('textbox', { name: 'Datasource Name' }).fill('Files DataSource');

      await page.getByRole('button', { name: 'Create' }).click();
      await page.waitForURL(/\/knowledge-bases\/\d+\/data-sources/);
      await pollKbOverviewUntill(page, kbId, overview => overview.documents.total === 1);
    });

    await test.step('Configure Web Pages Data Source', async () => {
      await page.getByRole('button', { name: 'Select Pages' }).click();
      await page.waitForURL(/data-sources\/new\?type=web_single_page/);

      await page.getByRole('button', { name: 'New Item' }).click();
      await page.getByPlaceholder('https://example.com').fill('http://static-web-server/example-doc-1.html');

      await page.getByRole('textbox', { name: 'Datasource Name' }).fill('Web Pages DataSource');

      await page.getByRole('button', { name: 'Create' }).click();
      await page.waitForURL(/\/knowledge-bases\/\d+\/data-sources/);
      await pollKbOverviewUntill(page, kbId, overview => overview.documents.total === 2);

      // Check document exists
      await page.getByRole('button', { name: /^Documents/ }).click();
      await expect(page.getByRole('button', { name: 'Example Document 1' })).toBeVisible();
      await page.getByRole('button', { name: /^Data Sources/ }).click();

      await page.waitForURL(/\/knowledge-bases\/\d+\/data-sources/);
    });

    await test.step('Configure Sitemap Data Source', async () => {
      await page.getByRole('button', { name: 'Select web sitemap.' }).click();
      await page.waitForURL(/data-sources\/new\?type=web_sitemap/);

      await page.getByRole('textbox', { name: 'Sitemap URL' }).fill('http://static-web-server/example-sitemap.xml');

      await page.getByRole('textbox', { name: 'Datasource Name' }).fill('Web Sitemap DataSource');

      await page.getByRole('button', { name: 'Create' }).click();
      await page.waitForURL(/\/knowledge-bases\/\d+\/data-sources/);
      await pollKbOverviewUntill(page, kbId, overview => overview.documents.total === 4);

      // Check document exists
      await page.getByRole('button', { name: /^Documents/ }).click();
      await expect(page.getByRole('button', { name: 'Example Document 1' })).toHaveCount(2); // Documents are not deduplicated.
      await expect(page.getByRole('button', { name: 'Example Document 2' })).toBeVisible();
      await page.getByRole('button', { name: /^Data Sources/ }).click();
    });

    test.slow();
    await test.step('Check for index progress', async () => {
      await pollKbOverviewUntill(page, kbId,
        overview =>
          overview.documents.total === 4
          && overview.chunks.total === 4
          && overview.vector_index.completed === 4
          && overview.kg_index.completed === 4,
      );
    });
  });

  test('Delete Data Sources', async ({ page }) => {
    const kbId = await createFeaturedKnowledgeBase(page, 'KnowledgeBase 2');
    await configureSimpleDataSource(page, kbId);

    await test.step('Delete Document', async () => {
      await page.goto(`/knowledge-bases/${kbId}/data-sources`);
      await page.getByRole('button', { name: 'Delete' }).click();
      await page.getByRole('button', { name: 'Continue' }).click();
      await page.getByRole('button', { name: 'Continue' }).waitFor({ state: 'detached' });
      // FIXME: reload data sources after deletion
      await pollKbOverviewUntill(page, kbId, overview => {
        return overview.documents.total === 0 && overview.chunks.total === 0;
      });
    });

    await test.step('Wait for documents and chunks to be deleted', async () => {
      await pollKbOverviewUntill(page, kbId, overview => {
        return overview.documents.total === 0 && overview.chunks.total === 0;
      });
    });
  });

  test('Delete Documents', async ({ page }) => {
    const kbId = await createFeaturedKnowledgeBase(page, 'KnowledgeBase 3');
    await configureSimpleDataSource(page, kbId);

    await test.step('Delete Document', async () => {
      await page.goto(`/knowledge-bases/${kbId}`);
      // FIXME: add aria roles
      await page.getByRole('button').filter({ has: page.locator('.lucide-ellipsis') }).click();
      await page.getByRole('menuitem', { name: 'Delete' }).click();
      await page.getByRole('button', { name: 'Continue' }).click();
      await page.getByRole('button', { name: 'Continue' }).waitFor({ state: 'detached' });

    });

    await test.step('Wait for documents and chunks to be deleted', async () => {
      await pollKbOverviewUntill(page, kbId, overview => {
        return overview.documents.total === 0 && overview.chunks.total === 0;
      });
    });
  });
});

async function createFeaturedKnowledgeBase (page: Page, name: string, enableKnowledgeGraph = false) {
  await loginViaApi(page);
  return await test.step(`Create KnowledgeBase ${name} (kg_index ${enableKnowledgeGraph ? 'enabled' : 'disabled'})`, async () => {
    await test.step('Navigate to Create KnowledgeBase Page', async () => {
      await page.goto('/knowledge-bases');
      await page.getByRole('button', { name: 'New Knowledge Base' }).click();
      await page.waitForURL('/knowledge-bases/new');
    });

    await test.step('Fill KnowledgeBase Form', async () => {
      await page.getByRole('textbox', { name: 'Name' }).fill(name);
      await page.getByRole('textbox', { name: 'Description' }).fill(`KnowledgeBase Description for ${name}`);

      await selectOption(page, 'LLM', /My LLM/);
      await selectOption(page, 'Embedding Model', /My Embedding Model/);

      if (enableKnowledgeGraph) {
        await turnSwitch(page, 'Knowledge Graph Index');
      }
    });

    return await test.step('Create and jump to data sources page', async () => {
      await page.getByRole('button', { name: 'Create' }).click();
      await page.waitForURL(/\/knowledge-bases\/\d+\/data-sources/);

      const [, idString] = /\/knowledge-bases\/(\d+)\/data-sources/.exec(page.url());
      return parseInt(idString);
    });
  });
}

async function configureSimpleDataSource (page: Page, kbId: number, enableKnowledgeGraph = false) {
  await test.step(`Configure simple data source`, async () => {
    await test.step(`Upload simple file`, async () => {
      await page.getByRole('button', { name: 'Upload Files' }).click();
      await page.waitForURL(/data-sources\/new\?type=file/);

      await page.setInputFiles('[name=files]', 'res/sample.pdf');

      await page.getByRole('textbox', { name: 'Datasource Name' }).fill('Simple DataSource');

      await page.getByRole('button', { name: 'Create' }).click();
      await page.waitForURL(/\/knowledge-bases\/\d+\/data-sources/);

    });

    await test.step(`Wait for index progress`, async () => {
      await pollKbOverviewUntill(page, kbId, overview => {
        expect(overview.documents.total).toBe(1);
        return !!overview.vector_index.completed && (!enableKnowledgeGraph || !!overview.kg_index.completed);
      });
    });
    return kbId;
  });
}

async function pollKbOverviewUntill (page: Page, kbId: number, isOk: (json: any) => boolean) {
  await test.step('Poll kb overview api', async () => {
    let i = 0;
    while (true) {
      const ok = await test.step(`Poll rounds ${++i}`, async () => {
        await page.waitForTimeout(500);
        const response = await page.request.get(`/api/v1/admin/knowledge_bases/${kbId}/overview`);
        expect(response.ok()).toBe(true);
        const overview = await response.json();
        return isOk(overview);
      });
      if (ok) {
        break;
      }
    }
  });
}
