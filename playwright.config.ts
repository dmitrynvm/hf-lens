import { defineConfig, devices } from '@playwright/test'

const PORT = 4174

// Visual regression of the built app (vite preview). Desktop runs in both themes; the viewport is tall
// enough that the sidebar and main column show all their content without their own scrollbars.
export default defineConfig({
  testDir: 'e2e',
  testMatch: /\.e2e\.ts$/,
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  reporter: [['list'], ['html', { open: 'never' }]],
  snapshotPathTemplate: '{testDir}/__screenshots__/{projectName}/{arg}{ext}',
  // threshold: per-pixel colour distance. The default 0.2 lets pale washes (e.g. --now-wash vs --paper) pass as equal.
  expect: { toHaveScreenshot: { threshold: 0.05, maxDiffPixelRatio: 0.001 } },
  use: {
    baseURL: `http://127.0.0.1:${PORT}`,
    ...devices['Desktop Chrome'],
    deviceScaleFactor: 1,
    locale: 'en-GB',
    timezoneId: 'UTC',
  },
  projects: [
    { name: 'light', testIgnore: /mobile/, use: { colorScheme: 'light', viewport: { width: 1440, height: 2300 } } },
    { name: 'dark', testIgnore: /mobile/, use: { colorScheme: 'dark', viewport: { width: 1440, height: 2300 } } },
    { name: 'mobile', testMatch: /mobile\.e2e\.ts$/, use: { colorScheme: 'light', viewport: { width: 375, height: 812 }, hasTouch: true } },
  ],
  webServer: {
    command: `npm run build && npx vite preview --host 127.0.0.1 --port ${PORT} --strictPort`,
    url: `http://127.0.0.1:${PORT}`,
    reuseExistingServer: false,
    timeout: 120_000,
  },
})
