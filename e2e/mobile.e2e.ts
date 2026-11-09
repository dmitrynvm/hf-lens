import { expect, test } from '@playwright/test'
import { openApp, step } from './fixtures'

test('narrow layout stacks the sidebar above the tabs', async ({ page }) => {
  await openApp(page)
  await step(page, 5)
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(375)
  await expect(page).toHaveScreenshot('block.png', { fullPage: true })
})

test('network view scrolls sideways instead of widening the page', async ({ page }) => {
  await openApp(page, 'model=120b&view=network&layer=0')
  await step(page, 5)
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(375)
  await expect(page).toHaveScreenshot('network.png', { fullPage: true })
})
