// pages/_app.tsx
import '@/styles/globals.css';
import type { AppProps } from 'next/app';
import { useRouter } from 'next/router';
import { useEffect } from 'react';

export default function App({ Component, pageProps }: AppProps) {
  const router = useRouter();

  useEffect(() => {
    // Check authentication on route change
    if (router.pathname !== '/login') {
      const token = localStorage.getItem('quantum_token');
      if (!token) {
        router.push('/login');
      }
    }
  }, [router.pathname]);

  return <Component {...pageProps} />;
}
