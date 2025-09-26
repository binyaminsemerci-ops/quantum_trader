import React from 'react';

export type StatusCardProps = {
  title?: string;
  status?: 'ok' | 'warn' | 'error';
};

export default function StatusCard({ title = 'Status', status = 'ok' }: StatusCardProps) {
  const color = status === 'ok' ? 'green' : status === 'warn' ? 'yellow' : 'red';
  return (
    <div data-testid="status-card" style={{ border: `2px solid ${color}`, padding: 8 }}>
      <div>{title}</div>
      <div>{status}</div>
    </div>
  );
}
// ...existing simple StatusCard component kept above
