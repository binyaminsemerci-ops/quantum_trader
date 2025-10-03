import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import type { Plugin } from 'vite'

// Debug plugin to log precise listen lifecycle & attempt external self-connect.
const debugBindPlugin: Plugin = {
  name: 'debug-bind-plugin',
  configureServer(server) {
    const httpServer = server.httpServer
    if (!httpServer) return
    httpServer.on('listening', () => {
      const addr = httpServer.address()
      console.log('[vite-debug] httpServer listening address=', addr)
      // Attempt delayed loopback connect via net (import dynamically to avoid bundling confusion)
      setTimeout(async () => {
        try {
          const net = await import('node:net')
          const port = (addr as any)?.port
          if (!port) {
            console.log('[vite-debug] no port resolved from address object')
            return
          }
          const socket = net.createConnection({ port, host: '127.0.0.1' }, () => {
            console.log('[vite-debug] self external connect SUCCESS to port', port)
            socket.end()
          })
          socket.on('error', (e: any) => {
            console.log('[vite-debug] self external connect ERROR', e?.code, e?.message)
          })
        } catch (e) {
          console.log('[vite-debug] self connect attempt threw', e)
        }
      }, 250)
    })
    httpServer.on('error', (err: any) => {
      console.log('[vite-debug] httpServer error event', err?.code, err?.message)
    })
    process.on('uncaughtException', (e) => {
      console.log('[vite-debug] uncaughtException', e)
    })
    process.on('unhandledRejection', (e) => {
      console.log('[vite-debug] unhandledRejection', e)
    })
  }
}

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react(), debugBindPlugin],
  server: {
    host: '127.0.0.1',
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, '')
      }
    }
  },
})
