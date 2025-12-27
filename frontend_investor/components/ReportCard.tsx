// components/ReportCard.tsx
interface ReportCardProps {
  title: string;
  description: string;
  format: 'json' | 'csv' | 'pdf';
  downloadUrl: string;
  icon?: string;
}

export default function ReportCard({ title, description, format, downloadUrl, icon }: ReportCardProps) {
  const getFormatColor = () => {
    switch (format) {
      case 'json':
        return 'bg-blue-900/20 text-blue-400 border-blue-500/50';
      case 'csv':
        return 'bg-green-900/20 text-green-400 border-green-500/50';
      case 'pdf':
        return 'bg-red-900/20 text-red-400 border-red-500/50';
      default:
        return 'bg-quantum-card text-quantum-text border-quantum-border';
    }
  };

  const handleDownload = async () => {
    try {
      const token = localStorage.getItem('quantum_token');
      const response = await fetch(downloadUrl, {
        headers: {
          'Authorization': `Bearer ${token}`,
        },
      });
      
      if (!response.ok) throw new Error('Download failed');
      
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `quantumfond_report_${Date.now()}.${format}`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      window.URL.revokeObjectURL(url);
    } catch (error) {
      console.error('Download error:', error);
      alert('Failed to download report. Please try again.');
    }
  };

  return (
    <div className="bg-quantum-card border border-quantum-border rounded-lg p-5 hover:border-quantum-accent transition">
      <div className="flex items-start justify-between mb-3">
        <div className="flex-1">
          <h3 className="text-lg font-semibold text-quantum-text mb-1">
            {icon && <span className="mr-2">{icon}</span>}
            {title}
          </h3>
          <p className="text-sm text-quantum-muted">{description}</p>
        </div>
        <span className={`px-3 py-1 rounded-full text-xs font-medium border ${getFormatColor()}`}>
          {format.toUpperCase()}
        </span>
      </div>
      <button
        onClick={handleDownload}
        className="w-full mt-4 px-4 py-2 bg-quantum-accent hover:bg-green-600 text-white rounded-lg font-medium transition"
      >
        Download {format.toUpperCase()}
      </button>
    </div>
  );
}
