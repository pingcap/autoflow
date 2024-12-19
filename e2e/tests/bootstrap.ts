import { expect, test } from '@playwright/test';

test.use({
  trace: !!process.env.CI ? 'off' : 'on',
});

test('Bootstrap', async ({ browser, page }) => {
  test.slow();

  const {
    USERNAME,
    PASSWORD,
    E2E_LLM_PROVIDER,
    E2E_LLM_MODEL,
    E2E_LLM_CREDENTIALS,
    E2E_EMBEDDING_PROVIDER,
    E2E_EMBEDDING_MODEL,
    E2E_EMBEDDING_CREDENTIALS,
    E2E_RERANKER_PROVIDER,
    E2E_RERANKER_MODEL,
    E2E_RERANKER_CREDENTIALS,
  } = process.env;

  await test.step('Visit home page', async () => {
    await page.goto('/');

    // IMPORTANT: Prevent recording credentials
    await page.addStyleTag({
      content: `[name=credentials] { filter: blur(1.5rem); }`,
    });
    await expect(page).toHaveTitle('TiDB.AI');
    await expect(page.getByText('Ask anything about TiDB')).toBeVisible();
  });

  const hasWizardAlert = await page.getByText('This site is not ready to use yet.').isVisible();

  if (!hasWizardAlert) {
    return;
  }

  await test.step('Login', async () => {
    if (await page.getByRole('link', { name: 'Login', exact: true }).count() === 0) {
      console.warn('Already logged in');
      return;
    }
    await page.getByRole('link', { name: 'Login', exact: true }).click();

    const usernameInput = await page.waitForSelector('[name=username]');
    const passwordInput = await page.waitForSelector('[name=password]');
    const loginButton = page.getByRole('button', { name: 'Login', exact: true });

    // Fill in credentials
    await usernameInput.fill(USERNAME);
    await passwordInput.fill(PASSWORD);

    // Click login
    await loginButton.click();

    // Wait for dialog dismiss
    await page.getByRole('dialog', { name: 'Sign In' }).waitFor({ state: 'detached' });

    // Wait login
    await page.getByText(USERNAME).waitFor({ state: 'visible' });
  });

  await test.step('Open admin side menu', async () => {
    const modelTab = page.getByText('Models', { exact: true }).and(page.locator('[data-sidebar="menu-button"]'));
    if ((await modelTab.getAttribute('data-state')) !== 'open') {
      await modelTab.click();
    }
  });

  async function clickTab (text: string, url: string) {
    await test.step(`Goto ${text} page`, async () => {
      await page.getByText(text, { exact: true }).and(page.locator('[data-sidebar="menu-sub-button"]').or(page.locator('[data-sidebar="menu-button"]'))).click();
      await page.waitForURL(url);
      await page.getByText(`New ${text.replace(/s$/, '')}`).waitFor({ state: 'visible' });
    });
  }

  // Setup reranker
  await test.step(`Create Default Reranker (${E2E_RERANKER_PROVIDER} ${E2E_RERANKER_MODEL})`, async () => {
    await clickTab('Reranker Models', '/reranker-models');

    await page.getByText('Loading Data').waitFor({ state: 'detached' });
    if (await page.getByText('My Reranker').count() === 0) {
      await page.getByText('New Reranker Model').click();

      // Fill name
      const nameInput = await page.waitForSelector('[name=name]');
      await nameInput.fill('My Reranker');

      // Select provider
      await page.getByLabel('Provider').locator('..').locator('button').click();
      await page.getByRole('option').filter({
        has: page.getByText(E2E_RERANKER_PROVIDER, { exact: true }),
      }).click();

      // Fill model if provided
      if (E2E_RERANKER_MODEL) {
        const modelInput = await page.waitForSelector('[name=model]');
        await modelInput.fill(E2E_RERANKER_MODEL);
      }

      // Fill credentials
      if (E2E_RERANKER_CREDENTIALS) {
        const credentialsInput = await page.waitForSelector('[name=credentials]');
        await credentialsInput.fill(E2E_RERANKER_CREDENTIALS);
      }

      // Click create button
      const createButton = page.getByRole('button', { name: 'Create Reranker' });
      await createButton.scrollIntoViewIfNeeded();
      await createButton.click();

      // Wait for finish by check the url changes
      await page.waitForURL(/\/reranker-models\/\d+/);
    }
  });

  await test.step(`Create Default LLM (${E2E_LLM_PROVIDER} ${E2E_LLM_MODEL})`, async () => {
    await clickTab('LLMs', '/llms');

    await page.getByText('Loading Data').waitFor({ state: 'detached' });
    if (await page.getByText('My LLM').count() === 0) {
      await page.getByText('New LLM').click();

      // Fill name
      const nameInput = await page.waitForSelector('[name=name]');
      await nameInput.fill('My LLM');

      // Select provider
      await page.getByLabel('Provider').locator('..').locator('button').click();
      await page.getByRole('option').filter({
        has: page.getByText(E2E_LLM_PROVIDER, { exact: true }),
      }).click();

      // Fill model if provided
      if (E2E_LLM_MODEL) {
        const modelInput = await page.waitForSelector('[name=model]');
        await modelInput.fill(E2E_LLM_MODEL);
      }

      // Fill credentials
      const credentialsInput = await page.waitForSelector('[name=credentials]');
      await credentialsInput.fill(E2E_LLM_CREDENTIALS);

      // Click create button
      const createButton = page.getByRole('button', { name: 'Create LLM' });
      await createButton.scrollIntoViewIfNeeded();
      await createButton.click();

      // Wait for finish by check the url changes
      await page.waitForURL(/\/llms\/\d+/);
    }
  });

  await test.step(`Create Default Embedding model (${E2E_EMBEDDING_PROVIDER} ${E2E_EMBEDDING_MODEL || 'default'})`, async () => {
    await clickTab('Embedding Models', '/embedding-models');

    await page.getByText('Loading Data').waitFor({ state: 'detached' });
    if (await page.getByText('My Embedding Model').count() === 0) {
      await page.getByText('New Embedding Model').click();

      // Fill name
      const nameInput = await page.waitForSelector('[name=name]');
      await nameInput.fill('My Embedding Model');

      // Select provider
      await page.getByLabel('Provider').locator('..').locator('button').click();
      await page.getByRole('option').filter({
        has: page.getByText(E2E_EMBEDDING_PROVIDER, { exact: true }),
      }).click();

      // Fill model if provided
      if (E2E_EMBEDDING_MODEL) {
        const modelInput = await page.waitForSelector('[name=model]');
        await modelInput.fill(E2E_EMBEDDING_MODEL);
      }

      // Fill credentials
      const credentialsInput = await page.waitForSelector('[name=credentials]');
      await credentialsInput.fill(E2E_EMBEDDING_CREDENTIALS);

      const vectorDimensionInput = await page.waitForSelector('[name=vector_dimension]');
      await vectorDimensionInput.fill('1536');

      // Click create button
      const createButton = page.getByRole('button', { name: 'Create Embedding Model' });
      await createButton.scrollIntoViewIfNeeded();
      await createButton.click();

      // Wait for finish by check the url changes
      await page.waitForURL(/\/embedding-models\/\d+/);
    }
  });

  // Create Knowledge Base
  await test.step('Create Knowledge Base', async () => {
    await clickTab('Knowledge Bases', '/knowledge-bases');

    await page.getByText('Loading Data').waitFor({ state: 'detached' });
    if (await page.getByText('My Knowledge Base').count() === 0) {
      await page.getByText('New Knowledge Base').click();
      await page.waitForSelector('[name=name]');
      await page.fill('input[name=name]', 'My Knowledge Base');
      await page.fill('textarea[name=description]', 'This is E2E Knowledge Base.');
      await page.getByRole('button', { name: 'Submit', exact: true }).click();

      await page.waitForURL(/\/knowledge-bases\/1\/data-sources/);
    }

    // Create Datasource
    await test.step('Create Datasource', async () => {
      await page.goto('/knowledge-bases/1/data-sources');

      if (await page.getByText('sample.pdf').count() === 0) {
        await page.getByRole('button', { name: 'Files' }).click();

        const nameInput = await page.waitForSelector('[name=name]');
        await nameInput.fill('sample.pdf');

        await page.setInputFiles('[name=files]', 'sample.pdf');

        const createButton = page.getByRole('button', { name: 'Create' });
        await createButton.scrollIntoViewIfNeeded();

        await createButton.click();

        // Jump back to KB data source page
        await page.waitForURL(/\/knowledge-bases\/1\/data-sources$/);
      }
    });
  });

  // Update default Chat Engine
  await test.step('Update Chat Engine', async () => {
    await clickTab('Chat Engines', '/chat-engines');
    await page.getByText('Loading Data').waitFor({ state: 'detached' });
    await page.getByRole('link', { name: 'default' }).click();

    await page.getByRole('tab', { name: 'Retrieval' }).click();
    await page.getByRole('button', { name: 'Knowledge Base', exact: true }).click();
    await page.getByRole('option').filter({ has: page.getByText('My Knowledge Base') }).click();

    await page.getByRole('button', { name: 'Save', exact: true }).click();
    await page.getByRole('button', { name: 'Save', exact: true }).waitFor({ state: 'detached' });
  });

  await test.step('Reload and check wizard alert', async () => {
    await page.goto('/');
    await page.getByText('This site is not ready to use yet.').waitFor({ state: 'detached' });
  });

  await test.step('Documents count greater than 0', async () => {
    await page.goto('/knowledge-bases/1');
    await page.getByRole('button', { name: 'sample.pdf' }).waitFor({ state: 'visible' });
  });

  await test.step('Wait for indexing', async () => {
    while (true) {
      const response = await page.request.get('/api/v1/admin/knowledge_bases/1/overview');
      if (!response.ok()) {
        console.warn(`${response.status()} ${response.statusText()}`, await response.text());
      } else {
        const json = await response.json();
        if (json.vector_index.completed > 0) {
          break;
        }
      }
      await page.waitForTimeout(500);
    }
  });
});
