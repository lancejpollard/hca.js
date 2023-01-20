import TilingDisplay from './tst/Tiling'
import React from 'react'
import { createRoot } from 'react-dom/client'

const reactDiv = document.querySelector('#react')
assertElement(reactDiv)
const reactBase = createRoot(reactDiv)

reactBase.render(<TilingDisplay data={tiling} />)

function assertElement(obj: object | null): asserts obj is HTMLElement {
  if (obj instanceof HTMLElement) {
    return
  }

  throw new Error('Not an HTML element')
}
