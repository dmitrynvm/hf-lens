import { expect, test } from '@playwright/test'
import { openApp } from './fixtures'

// Behaviour, not looks: one theme is enough.
test.skip(({ colorScheme }) => colorScheme !== 'light')

test('a link carries model, tab and layer into a fresh page', async ({ page, context }) => {
  await openApp(page, 'model=20b&view=model&layer=0')
  await page.locator('.bead').nth(6).click()
  await expect(page).toHaveURL(/#model=20b&view=model&layer=5$/)

  const fresh = await context.newPage()
  await openApp(fresh, new URL(page.url()).hash.slice(1))
  await expect(fresh.locator('.brand')).toHaveText('openai/gpt-oss-20b')
  await expect(fresh.locator('#panel-model')).toBeVisible()
  await expect(fresh.getByRole('slider', { name: 'Layer' })).toHaveAttribute('aria-valuetext', /^L5 · /)
})

test('old seed and trace params are dropped from the link', async ({ page }) => {
  await openApp(page, 'model=20b&view=model&layer=0&seed=9&token=1&trace=AQMyMGI')
  await expect(page).toHaveURL(/#model=20b&view=model&layer=0$/)
})

test('the embedding and lm_head are selectable and survive the link', async ({ page }) => {
  await openApp(page, 'model=120b&view=network&layer=5')
  const scroller = page.locator('.diagram-scroll')
  const before = await scroller.evaluate((el) => el.scrollLeft)
  await scroller.getByRole('button', { name: /^L6 / }).click()
  await expect(page).toHaveURL(/&layer=6$/)
  expect(await scroller.evaluate((el) => el.scrollLeft)).toBeGreaterThan(before)

  await page.locator('.net-group.embed').click()
  await expect(page).toHaveURL(/&layer=embed$/)
  await expect(page.getByRole('slider', { name: 'Layer' })).toHaveAttribute('aria-valuetext', 'In · embedding')
  await expect(page.locator('.bead.selected')).toHaveCount(1)

  await openApp(page, 'model=120b&view=block&layer=head')
  await expect(page.getByRole('slider', { name: 'Layer' })).toHaveAttribute('aria-valuetext', 'Out · lm_head')
  await page.locator('.bead').nth(4).click()
  await expect(page).toHaveURL(/&layer=3$/)
  await expect(page.getByRole('slider', { name: 'Layer' })).toHaveAttribute('aria-valuetext', /^L3 · /)
})

test('loads Kimi-K3 from the model hash parameter', async ({ page }) => {
  await openApp(page, 'model=kimi-k3&view=network&layer=0')
  await expect(page.locator('.brand')).toHaveText('moonshotai/Kimi-K3')
  await expect(page.locator('#panel-network')).toBeVisible()
  await expect(page.getByRole('slider', { name: 'Layer' })).toHaveAttribute('aria-valuetext', /^L0 · /)
})
