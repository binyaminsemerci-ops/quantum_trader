// pages/login.tsx
import { useState, FormEvent } from 'react';
import { useRouter } from 'next/router';
import { useAuth } from '@/hooks/useAuth';

export default function Login() {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const router = useRouter();
  const { login } = useAuth();

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    try {
      const success = await login(username, password);
      if (success) {
        router.push('/');
      } else {
        setError('Invalid credentials. Please try again.');
      }
    } catch (err) {
      setError('Login failed. Please check your credentials.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-quantum-bg px-4">
      <div className="w-full max-w-md space-y-8">
        {/* Logo/Header */}
        <div className="text-center">
          <h1 className="text-4xl font-bold text-quantum-accent mb-2">
            QuantumFond
          </h1>
          <p className="text-quantum-muted text-lg">Investor Portal</p>
        </div>

        {/* Login Form */}
        <div className="bg-quantum-card border border-quantum-border rounded-lg p-8 shadow-2xl">
          <h2 className="text-2xl font-semibold text-center mb-6">
            Investor Login
          </h2>

          {error && (
            <div className="mb-4 p-3 bg-red-900/20 border border-red-500/50 rounded text-red-400 text-sm">
              {error}
            </div>
          )}

          <form onSubmit={handleSubmit} className="space-y-5">
            <div>
              <label htmlFor="username" className="block text-sm font-medium text-quantum-text mb-2">
                Username
              </label>
              <input
                id="username"
                type="text"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                placeholder="Enter your username"
                className="w-full px-4 py-3 bg-quantum-dark border border-quantum-border rounded-lg text-quantum-text placeholder-quantum-muted focus:outline-none focus:ring-2 focus:ring-quantum-accent focus:border-transparent transition"
                required
                disabled={loading}
              />
            </div>

            <div>
              <label htmlFor="password" className="block text-sm font-medium text-quantum-text mb-2">
                Password
              </label>
              <input
                id="password"
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="Enter your password"
                className="w-full px-4 py-3 bg-quantum-dark border border-quantum-border rounded-lg text-quantum-text placeholder-quantum-muted focus:outline-none focus:ring-2 focus:ring-quantum-accent focus:border-transparent transition"
                required
                disabled={loading}
              />
            </div>

            <button
              type="submit"
              disabled={loading}
              className="w-full py-3 bg-quantum-accent hover:bg-green-600 text-white font-semibold rounded-lg transition disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading ? 'Logging in...' : 'Login'}
            </button>
          </form>

          <div className="mt-6 text-center text-sm text-quantum-muted">
            <p>Read-only access for verified investors</p>
          </div>
        </div>

        {/* Footer */}
        <div className="text-center text-xs text-quantum-muted">
          <p>Secure access powered by QuantumFond Authentication</p>
          <p className="mt-1">© 2025 QuantumFond. All rights reserved.</p>
        </div>
      </div>
    </div>
  );
}
