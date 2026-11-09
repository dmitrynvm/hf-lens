/** 16 px control glyphs drawn in the button's text color. */
const ICONS = {
  back: <><path d="M3 3h2v10H3z" /><path d="M13 3v10L6 8z" /></>,
  play: <path d="M4 2.5v11L13.5 8z" />,
  pause: <><path d="M4 3h3v10H4z" /><path d="M9 3h3v10H9z" /></>,
  forward: <><path d="M11 3h2v10h-2z" /><path d="M3 3v10l7-5z" /></>,
  reset: <><path d="M12.5 9A4.5 4.5 0 1 1 8 3.5" fill="none" stroke="currentColor" strokeWidth="2" /><path d="M8 0.5v6L11.5 3.5z" /></>,
  add: <><path d="M7 2h2v12H7z" /><path d="M2 7h12v2H2z" /></>,
}

export function Icon({ name }: { name: keyof typeof ICONS }) {
  return <svg className="button-icon" viewBox="0 0 16 16" width="16" height="16" fill="currentColor" aria-hidden="true">{ICONS[name]}</svg>
}
