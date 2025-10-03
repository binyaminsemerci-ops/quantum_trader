// Enhanced version - full-featured HTTP polling dashboard with all components
import './App.css';
import FullSizeDashboard from './FullSizeDashboard';
import ErrorBoundary from './components/ErrorBoundary';

export default function App(): JSX.Element {
  return (
    <ErrorBoundary>
      <FullSizeDashboard />
    </ErrorBoundary>
  );
}
