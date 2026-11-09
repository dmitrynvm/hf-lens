export type MainView = 'model' | 'network' | 'block' | 'attention' | 'config'

/** Main-column tabs with their keyboard shortcuts. Ids are URL `view=` values, kept stable across label changes. */
export const MAIN_VIEWS: { id: MainView; label: string; key: string }[] = [
  { id: 'network', label: 'Model', key: '1' },
  { id: 'block', label: 'Block', key: '2' },
  { id: 'attention', label: 'Attention', key: '3' },
  { id: 'model', label: 'Params', key: '4' },
  { id: 'config', label: 'Config', key: '5' },
]

export const isMainView = (s: string | null): s is MainView => MAIN_VIEWS.some((v) => v.id === s)
