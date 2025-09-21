import React from 'react';

type TradingFormProps = {
  onStart?: () => void;
};

export default function TradingForm({ onStart }: TradingFormProps): JSX.Element {
  return (
    <div className="trading-form-placeholder">
      <h3>Trading Form (placeholder)</h3>
      <p>This component will be migrated incrementally.</p>
      <button type="button" onClick={() => onStart?.()}>Start</button>
    </div>
  );

}
