// pages/reports.tsx
import InvestorNavbar from '@/components/InvestorNavbar';
import ReportCard from '@/components/ReportCard';

const reports = [
  {
    title: 'Trade Journal Export',
    description: 'Complete trading history with entry/exit prices, P&L, and confidence scores',
    format: 'json' as const,
    icon: '📄',
  },
  {
    title: 'Performance Spreadsheet',
    description: 'Detailed performance metrics suitable for Excel analysis',
    format: 'csv' as const,
    icon: '📊',
  },
  {
    title: 'Formatted PDF Report',
    description: 'Professional report with charts and risk metrics for presentations',
    format: 'pdf' as const,
    icon: '📑',
  },
];

export default function Reports() {
  const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'https://api.quantumfond.com';

  return (
    <div className="min-h-screen bg-quantum-bg">
      <InvestorNavbar />
      
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-quantum-text mb-2">
            Download Reports
          </h1>
          <p className="text-quantum-muted">
            Export fund performance data in multiple formats
          </p>
        </div>

        {/* Report Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
          {reports.map((report) => (
            <ReportCard
              key={report.format}
              title={report.title}
              description={report.description}
              format={report.format}
              downloadUrl={`${apiUrl}/reports/export/${report.format}`}
              icon={report.icon}
            />
          ))}
        </div>

        {/* Additional Info */}
        <div className="bg-quantum-card border border-quantum-border rounded-lg p-6">
          <h3 className="text-lg font-semibold text-quantum-text mb-4">
            📚 Report Information
          </h3>
          <div className="space-y-4 text-sm">
            <div>
              <h4 className="text-quantum-accent font-medium mb-2">JSON Format</h4>
              <p className="text-quantum-muted">
                Machine-readable format suitable for API integration, data analysis, and custom reporting tools.
                Includes all raw data with full precision.
              </p>
            </div>
            <div>
              <h4 className="text-quantum-accent font-medium mb-2">CSV Format</h4>
              <p className="text-quantum-muted">
                Spreadsheet-compatible format that can be opened in Excel, Google Sheets, or other data analysis tools.
                Perfect for custom calculations and pivot tables.
              </p>
            </div>
            <div>
              <h4 className="text-quantum-accent font-medium mb-2">PDF Format</h4>
              <p className="text-quantum-muted">
                Professional formatted report with tables and styling. Ideal for presentations, regulatory submissions,
                and investor communications.
              </p>
            </div>
          </div>
        </div>

        {/* Report Schedule Info */}
        <div className="mt-6 bg-quantum-dark border border-quantum-border rounded-lg p-6">
          <h3 className="text-lg font-semibold text-quantum-text mb-4">
            📅 Reporting Schedule
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 text-sm">
            <div className="text-center">
              <div className="text-quantum-accent text-2xl mb-2">📈</div>
              <h4 className="font-medium text-quantum-text mb-1">Daily Updates</h4>
              <p className="text-quantum-muted">
                Performance metrics updated in real-time throughout trading hours
              </p>
            </div>
            <div className="text-center">
              <div className="text-quantum-accent text-2xl mb-2">📊</div>
              <h4 className="font-medium text-quantum-text mb-1">Weekly Reports</h4>
              <p className="text-quantum-muted">
                Comprehensive weekly performance summary sent every Monday
              </p>
            </div>
            <div className="text-center">
              <div className="text-quantum-accent text-2xl mb-2">📑</div>
              <h4 className="font-medium text-quantum-text mb-1">Monthly Analysis</h4>
              <p className="text-quantum-muted">
                Detailed monthly report with risk analysis and strategy insights
              </p>
            </div>
          </div>
        </div>

        {/* Support Section */}
        <div className="mt-6 text-center p-6 bg-quantum-card border border-quantum-border rounded-lg">
          <p className="text-quantum-muted">
            Need custom reports or have questions about the data?
          </p>
          <p className="text-quantum-accent mt-2">
            Contact our investor relations team at{' '}
            <a href="mailto:investors@quantumfond.com" className="underline hover:text-green-300">
              investors@quantumfond.com
            </a>
          </p>
        </div>
      </div>
    </div>
  );
}
