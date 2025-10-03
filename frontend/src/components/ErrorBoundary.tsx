import React from 'react';

interface ErrorBoundaryState {
  hasError: boolean;
  error?: Error;
  info?: React.ErrorInfo;
}

export class ErrorBoundary extends React.Component<React.PropsWithChildren, ErrorBoundaryState> {
  constructor(props: React.PropsWithChildren) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, info: React.ErrorInfo) {
    // eslint-disable-next-line no-console
    console.error('[ErrorBoundary] Caught error:', error, info);
    this.setState({ info });
  }

  handleReload = () => {
    this.setState({ hasError: false, error: undefined, info: undefined });
    window.location.reload();
  };

  render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen flex items-center justify-center bg-gray-900 text-gray-100 p-6">
          <div className="max-w-md w-full space-y-4">
            <h1 className="text-2xl font-bold text-red-400">Noe gikk galt (runtime error)</h1>
            <p className="text-sm text-gray-400">Dashboardet krasjet men vi fanget feilen slik at du slipper hvit skjerm.</p>
            {this.state.error && (
              <pre className="bg-gray-800 p-3 rounded text-xs overflow-auto max-h-48 border border-gray-700">
{this.state.error.message}
              </pre>
            )}
            <button
              onClick={this.handleReload}
              className="w-full bg-blue-600 hover:bg-blue-500 transition-colors rounded py-2 font-medium"
            >
              Last siden på nytt
            </button>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}

export default ErrorBoundary;