import puppeteer from 'puppeteer';
import fs from 'fs';

(async () => {
  const argPort = process.argv[2];
  const port = argPort || process.env.PORT || '5174';
  const url = `http://localhost:${port}/`;
  const out = `../artifacts/dev_screenshot_${port}.png`;
  try {
    const browser = await puppeteer.launch({ args: ['--no-sandbox', '--disable-setuid-sandbox'] });
    const page = await browser.newPage();
    await page.setViewport({ width: 1280, height: 900 });
    await page.goto(url, { waitUntil: 'networkidle0', timeout: 30000 });
    await page.screenshot({ path: out, fullPage: true });
    await browser.close();
    console.log('Saved screenshot to', out);
  } catch (err) {
    console.error('Failed to capture screenshot', err);
    process.exit(1);
  }
})();