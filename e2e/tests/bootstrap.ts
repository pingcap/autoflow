import { expect, test } from '@playwright/test';
import { submitAndWaitSavedByLabel } from '../utils/settings';

test.use({
  trace: !!process.env.CI ? 'off' : 'on',
});

test('Bootstrap', async ({ page }) => {
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
  });

  const hasWizardDialog = await test.step('Wizard Dialog', async () => {
    if ((await page.getByText('Almost there...').count()) === 0) {
      console.warn('Already bootstrapped.');
      return false;
    }

    return true;
  });

  if (!hasWizardDialog) {
    return;
  }

  await test.step('Login', async () => {
    if (await page.getByRole('button', { name: 'Login', exact: true }).count() === 0) {
      console.warn('Already logged in');
      return;
    }

    const usernameInput = await page.waitForSelector('[name=username]');
    const passwordInput = await page.waitForSelector('[name=password]');
    const loginButton = page.getByRole('button', { name: 'Login', exact: true });

    // Fill in credentials
    await usernameInput.fill(USERNAME);
    await passwordInput.fill(PASSWORD);

    // Click login
    await loginButton.click();

    // Wait login
    await page.getByText('Your app is not fully configured yet. Please complete the setup process.').waitFor();
  });

  // Setup reranker
  await test.step(`Create Default Reranker (${E2E_RERANKER_PROVIDER} ${E2E_RERANKER_MODEL})`, async () => {
    const header = page.getByText('Setup default Reranker');
    if (await header.locator('.lucide-circle-alert').count() === 0) {
      // Already configured.
      console.warn('Default Reranker already configured.');
    } else {
      await header.click();

      // Fill name
      const nameInput = await page.waitForSelector('[name=name]');
      await nameInput.fill('My Reranker');

      // Select provider
      await page.getByLabel('Provider').locator('..').locator('button').click();
      await page.getByText(E2E_RERANKER_PROVIDER, { exact: true }).click();

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

      // Toggle default switch
      // TODO: should enable by default and and readOnly
      const toggleDefaultRerankerSwitchButton = page.getByRole('switch');
      if ((await toggleDefaultRerankerSwitchButton.evaluate(node => node.getAttribute('aria-checked'))) !== 'true') {
        await toggleDefaultRerankerSwitchButton.click();
      }

      // Click create button
      const createButton = page.getByText('Create Reranker');
      await createButton.scrollIntoViewIfNeeded();
      await createButton.click();

      // Wait for finish by check the alert icon in header disappear
      try {
        await header.locator('.lucide-circle-alert').waitFor({ state: 'detached', timeout: 10000 });
      } catch (e) {
        throw new Error(await page.getByText('Failed to').locator('..').textContent({ timeout: 1000 }));
      }
    }
  });

  // Setup langfuse
  await test.step(`Setup langfuse`, async () => {
    const header = page.getByText('Setup Langfuse');
    if (await header.locator('.lucide-circle-alert').count() === 0) {
      // Already configured.
      console.warn('Langfuse already configured.');
    } else {
      await header.click();

      // Fill Info
      await page.getByLabel('Langfuse Public Key', { exact: true }).fill('lf_pk_1234567890');
      await submitAndWaitSavedByLabel(page, 'Langfuse Public Key');

      await page.getByLabel('Langfuse Secret Key', { exact: true }).fill('lf_sk_1234567890');
      await submitAndWaitSavedByLabel(page, 'Langfuse Secret Key');

      await page.getByLabel('Langfuse Host', { exact: true }).fill('http://localhost:5001');
      await submitAndWaitSavedByLabel(page, 'Langfuse Host');
    }
  });

  await test.step(`Create Default LLM (${E2E_LLM_PROVIDER} ${E2E_LLM_MODEL})`, async () => {
    const header = page.getByText('Setup default LLM');
    if (await header.locator('.lucide-circle-alert').count() === 0) {
      // Already configured.
      console.warn('Default LLM already configured.');
    } else {
      await header.click();

      // Fill name
      const nameInput = await page.waitForSelector('[name=name]');
      await nameInput.fill('My LLM');

      // Select provider
      await page.getByLabel('Provider').locator('..').locator('button').click();
      await page.getByText(E2E_LLM_PROVIDER, { exact: true }).click();

      // Fill model if provided
      if (E2E_LLM_MODEL) {
        const modelInput = await page.waitForSelector('[name=model]');
        await modelInput.fill(E2E_LLM_MODEL);
      }

      // Fill credentials
      const credentialsInput = await page.waitForSelector('[name=credentials]');
      await credentialsInput.fill(E2E_LLM_CREDENTIALS);

      // Toggle default switch
      // TODO: should enable by default and and readOnly
      const toggleDefaultLLMSwitchButton = page.getByRole('switch');
      if ((await toggleDefaultLLMSwitchButton.evaluate(node => node.getAttribute('aria-checked'))) !== 'true') {
        await toggleDefaultLLMSwitchButton.click();
      }

      // Click create button
      const createButton = page.getByText('Create LLM');
      await createButton.scrollIntoViewIfNeeded();
      await createButton.click();

      // Wait for finish by check the alert icon in header disappear
      try {
        await header.locator('.lucide-circle-alert').waitFor({ state: 'detached', timeout: 10000 });
      } catch (e) {
        throw new Error(await page.getByText('Failed to').locator('..').textContent({ timeout: 1000 }));
      }
    }
  });

  await test.step(`Create Default Embedding model (${E2E_EMBEDDING_PROVIDER} ${E2E_EMBEDDING_MODEL || 'default'})`, async () => {
    const header = page.getByText('Setup default Embedding Model');
    if (await header.locator('.lucide-circle-alert').count() === 0) {
      // Already configured.
      console.warn('Default Embedding Model already configured.');
    } else {
      await header.click();

      // Fill name
      const nameInput = await page.waitForSelector('[name=name]');
      await nameInput.fill('My Embedding Model');

      // Select provider
      await page.getByLabel('Provider').locator('..').locator('button').click();
      await page.getByText(E2E_EMBEDDING_PROVIDER, { exact: true }).click();

      // Fill model if provided
      if (E2E_EMBEDDING_MODEL) {
        const modelInput = await page.waitForSelector('[name=model]');
        await modelInput.fill(E2E_EMBEDDING_MODEL);
      }

      // Fill credentials
      const credentialsInput = await page.waitForSelector('[name=credentials]');
      await credentialsInput.fill(E2E_EMBEDDING_CREDENTIALS);

      // Click create button
      const createButton = page.getByText('Create Embedding Model');
      await createButton.scrollIntoViewIfNeeded();
      await createButton.click();

      // Wait for finish by check the alert icon in header disappear
      try {
        await header.locator('.lucide-circle-alert').waitFor({ state: 'detached', timeout: 10000 });
      } catch (e) {
        throw new Error(await page.getByText('Failed to').locator('..').textContent({ timeout: 1000 }));
      }
    }
  });

  // Create Datasource
  await test.step('Create Datasource', async () => {
    const header = page.getByText('Setup Datasource');
    if (await header.locator('.lucide-circle-alert').count() === 0) {
      // Already configured.
      console.warn('Datasource already configured.');
    } else {
      await header.click();

      const nameInput = await page.waitForSelector('[name=name]');
      await nameInput.fill('sample.pdf');

      const descriptionInput = await page.waitForSelector('textarea[name=description]');
      await descriptionInput.fill('This is sample.pdf.');

      await page.setInputFiles('[name=files]', 'sample.pdf');

      const createButton = page.getByText('Create Datasource');
      await createButton.scrollIntoViewIfNeeded();

      await createButton.click();

      await header.locator('.lucide-circle-alert').waitFor({ state: 'detached', timeout: 10000 });
    }
  });

  await test.step('Reload and check wizard dialog', async () => {
    await page.reload();
    await expect(page.getByText('Almost there...')).toHaveCount(0);
  });

  await test.step('Documents count greater than 0', async () => {
    await page.goto('/documents');
    await page.getByRole('link', { name: 'sample.pdf' }).waitFor({ state: 'visible' });
  });

  await test.step('Wait for indexing', async () => {
    while (true) {
      const response = await page.request.get('/api/v1/admin/rag/index-progress');
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
