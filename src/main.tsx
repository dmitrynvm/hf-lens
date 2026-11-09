import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import '@fontsource/barlow-semi-condensed/500.css'
import '@fontsource/barlow-semi-condensed/600.css'
import '@fontsource/public-sans/400.css'
import '@fontsource/public-sans/600.css'
import './index.css'
import App from './App.tsx'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
