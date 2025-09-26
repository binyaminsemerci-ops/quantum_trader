// React import not required with new JSX transform
import { render, screen, waitFor } from '@testing-library/react'
import '@testing-library/jest-dom'
import { vi } from 'vitest'

import PriceChart from '../PriceChart'

const mockCandles = Array.from({ length: 10 }).map((_, i) => ({
  time: new Date(Date.now() - (10 - i) * 60000).toISOString(),
  open: 100 + i,
  high: 101 + i,
  low: 99 + i,
  close: 100 + i,
  volume: 10 + i,
}))

beforeEach(() => {
  // Mock global fetch used by the component
  ;(globalThis as any).fetch = vi.fn(() =>
    Promise.resolve({ ok: true, json: () => Promise.resolve(mockCandles) })
  )
})

test('renders PriceChart and shows latest price', async () => {
  render(<PriceChart />)

  // Title should be present synchronously
  expect(screen.getByText(/Price chart/i)).toBeInTheDocument()

  // Latest price appears after fetch resolves
  await waitFor(() => {
    const latest = mockCandles[mockCandles.length - 1]
    expect(screen.getByText(new RegExp(String(latest.close).slice(0, 5)))).toBeInTheDocument()
  })
})
