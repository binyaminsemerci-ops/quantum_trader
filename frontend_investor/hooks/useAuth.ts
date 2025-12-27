// hooks/useAuth.ts
import { useEffect, useState } from 'react';
import { useRouter } from 'next/router';

export interface AuthUser {
  username: string;
  role: 'investor' | 'admin';
  access_token: string;
}

export function useAuth() {
  const [user, setUser] = useState<AuthUser | null>(null);
  const [loading, setLoading] = useState(true);
  const router = useRouter();

  useEffect(() => {
    checkAuth();
  }, []);

  const checkAuth = () => {
    const token = localStorage.getItem('quantum_token');
    const userData = localStorage.getItem('quantum_user');
    
    if (token && userData) {
      try {
        setUser(JSON.parse(userData));
      } catch (e) {
        logout();
      }
    }
    setLoading(false);
  };

  const login = async (username: string, password: string): Promise<boolean> => {
    try {
      const authUrl = process.env.NEXT_PUBLIC_AUTH_URL || 'https://auth.quantumfond.com';
      const response = await fetch(`${authUrl}/login`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ username, password }),
      });

      if (!response.ok) {
        throw new Error('Login failed');
      }

      const data = await response.json();
      
      localStorage.setItem('quantum_token', data.access_token);
      localStorage.setItem('quantum_user', JSON.stringify({
        username: data.username || username,
        role: data.role || 'investor',
        access_token: data.access_token,
      }));

      setUser({
        username: data.username || username,
        role: data.role || 'investor',
        access_token: data.access_token,
      });

      return true;
    } catch (error) {
      console.error('Login error:', error);
      return false;
    }
  };

  const logout = () => {
    localStorage.removeItem('quantum_token');
    localStorage.removeItem('quantum_user');
    setUser(null);
    router.push('/login');
  };

  const getToken = (): string | null => {
    return localStorage.getItem('quantum_token');
  };

  return { user, loading, login, logout, getToken, isAuthenticated: !!user };
}
