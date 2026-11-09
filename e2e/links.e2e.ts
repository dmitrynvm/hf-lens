import { expect, test } from '@playwright/test'
import { openApp } from './fixtures'

// Behaviour, not looks: one theme is enough.
test.skip(({ colorScheme }) => colorScheme !== 'light')

test('a link carries model, tab and layer into a fresh page', async ({ page, context }) => {
  await openApp(page, 'model=20b&view=routing&layer=0')
  await page.locator('.bead').nth(6).click()
  await expect(page).toHaveURL(/#model=20b&view=routing&layer=5$/)

  const fresh = await context.newPage()
  await openApp(fresh, new URL(page.url()).hash.slice(1))
  await expect(fresh.locator('.brand')).toHaveText('openai/gpt-oss-20b')
  await expect(fresh.locator('#panel-routing')).toBeVisible()
  await expect(fresh.getByRole('slider', { name: 'Layer' })).toHaveAttribute('aria-valuetext', /^L5 · /)
})

test('old seed and trace params are dropped from the link', async ({ page }) => {
  await openApp(page, 'model=20b&view=routing&layer=0&seed=9&token=1&trace=AQMyMGI')
  await expect(page).toHaveURL(/#model=20b&view=routing&layer=0$/)
})

test('the embedding and lm_head are selectable and survive the link', async ({ page }) => {
  await openApp(page, 'model=120b&view=network&layer=5')
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
