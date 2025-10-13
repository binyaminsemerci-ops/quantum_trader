import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
	plugins: [react()],
	server: {
		port: 5173,
		host: '0.0.0.0',
		strictPort: true,
		proxy: {
			'/stats': 'http://localhost:8080',
			'/trades': 'http://localhost:8080',
			'/trade_logs': 'http://localhost:8080',
			'/signals': 'http://localhost:8080',
			'/prices': 'http://localhost:8080',
			'/candles': 'http://localhost:8080',
		},
	},
	preview: {
		port: 5173,
	},
});
