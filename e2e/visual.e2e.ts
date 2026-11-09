import { expect, test } from '@playwright/test'
import { openApp, step } from './fixtures'

// The token sits on L3 in every shot: board row, heatmap row, band status and block experts all show it.
test.beforeEach(async ({ page }) => {
  await openApp(page)
  await step(page, 5)
  await expect(page.locator('.topbar-status')).toHaveText('L3 of 36')
})

for (const [key, view] of [['1', 'network'], ['2', 'block'], ['3', 'attention'], ['4', 'model'], ['5', 'config']] as const) {
  test(`${view} view`, async ({ page }) => {
    await page.locator('body').click({ position: { x: 1, y: 1 } }) // keys go to the page, not the Step button
    await page.keyboard.press(key)
    await expect(page.locator(`#panel-${view}`)).toBeVisible()
    await expect(page).toHaveScreenshot(`${view}.png`)
  })
}

test('keyboard focus stays inside the tab switcher', async ({ page }) => {
  await page.locator('#tab-block').focus()
  await page.keyboard.press('ArrowRight')
  await expect(page.locator('#tab-attention')).toBeFocused()
  await expect(page.locator('.tabs')).toHaveScreenshot('tabs-focus.png')
})
