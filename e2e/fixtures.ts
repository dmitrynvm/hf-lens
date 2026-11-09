import type { Page } from '@playwright/test'

/** Open the app on a fixed state, with the self-hosted fonts loaded so text renders identically every run. */
export async function openApp(page: Page, hash = 'model=120b&view=block&layer=0') {
  await page.goto(`/#${hash}`)
  await page.locator('.tabs').waitFor()
  await page.evaluate(() => document.fonts.ready)
}

/** Single-step the token flow: the first step is the embedding, then L0, L1, ... */
export async function step(page: Page, times: number) {
  const button = page.getByRole('button', { name: 'Step forward', exact: true })
  for (let i = 0; i < times; i++) await button.click()
}
