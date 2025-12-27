// components/InvestorNavbar.tsx
import { useRouter } from 'next/router';
import { useAuth } from '@/hooks/useAuth';

interface NavItem {
  icon: string;
  label: string;
  path: string;
}

const navItems: NavItem[] = [
  { icon: '🏠', label: 'Dashboard', path: '/' },
  { icon: '💼', label: 'Portfolio', path: '/portfolio' },
  { icon: '📈', label: 'Performance', path: '/performance' },
  { icon: '⚠️', label: 'Risk', path: '/risk' },
  { icon: '🤖', label: 'AI Models', path: '/models' },
  { icon: '📊', label: 'Reports', path: '/reports' },
];

export default function InvestorNavbar() {
  const router = useRouter();
  const { user, logout } = useAuth();

  return (
    <nav className="bg-quantum-dark border-b border-quantum-border">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <div className="flex items-center space-x-3">
            <div className="text-2xl font-bold text-quantum-accent">
              QuantumFond
            </div>
            <span className="text-quantum-muted text-sm">Investor Portal</span>
          </div>

          {/* Navigation Links */}
          <div className="hidden md:flex space-x-1">
            {navItems.map((item) => (
              <button
                key={item.path}
                onClick={() => router.push(item.path)}
                className={`px-4 py-2 rounded-lg text-sm font-medium transition ${
                  router.pathname === item.path
                    ? 'bg-quantum-accent text-white'
                    : 'text-quantum-muted hover:text-quantum-text hover:bg-quantum-card'
                }`}
              >
                <span className="mr-2">{item.icon}</span>
                {item.label}
              </button>
            ))}
          </div>

          {/* User Menu */}
          <div className="flex items-center space-x-4">
            <div className="text-sm text-quantum-muted">
              {user?.username}
            </div>
            <button
              onClick={logout}
              className="px-4 py-2 bg-red-900/20 hover:bg-red-900/30 text-red-400 rounded-lg text-sm font-medium transition"
            >
              Logout
            </button>
          </div>
        </div>
      </div>

      {/* Mobile Navigation */}
      <div className="md:hidden border-t border-quantum-border px-4 py-2 space-y-1">
        {navItems.map((item) => (
          <button
            key={item.path}
            onClick={() => router.push(item.path)}
            className={`w-full text-left px-3 py-2 rounded-lg text-sm font-medium transition ${
              router.pathname === item.path
                ? 'bg-quantum-accent text-white'
                : 'text-quantum-muted hover:text-quantum-text hover:bg-quantum-card'
            }`}
          >
            <span className="mr-2">{item.icon}</span>
            {item.label}
          </button>
        ))}
      </div>
    </nav>
  );
}
