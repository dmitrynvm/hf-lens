export type MainView = 'network' | 'block' | 'attention' | 'routing' | 'config'

/** Main-column tabs with their keyboard shortcuts. */
export const MAIN_VIEWS: { id: MainView; label: string; key: string }[] = [
  { id: 'network', label: 'Network', key: '1' },
  { id: 'block', label: 'Block', key: '2' },
  { id: 'attention', label: 'Attention', key: '3' },
  { id: 'routing', label: 'Routing', key: '4' },
  { id: 'config', label: 'Config', key: '5' },
]

export const isMainView = (s: string | null): s is MainView => MAIN_VIEWS.some((v) => v.id === s)
