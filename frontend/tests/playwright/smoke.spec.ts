import { test, expect } from '@playwright/test';

test('spa shell loads', async ({ page }) => {
  // Configure this to the dev server URL used locally (example: http://127.0.0.1:3001)
  await page.goto(process.env.PLAYWRIGHT_BASE_URL ?? 'http://127.0.0.1:3001/');
  // Assert that the root div or page title exists
  await expect(page.locator('#root')).toBeVisible({ timeout: 5000 }).catch(async () => {
    // fallback: check for a heading
    await expect(page.locator('h1')).toContainText('Quantum Trader', { timeout: 5000 });
  });
});
