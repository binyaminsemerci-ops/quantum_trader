// React import not required with jsx runtime
import { render, screen } from '@testing-library/react';
import StatusCard from '../components/StatusCard';

describe('StatusCard', () => {
  it('renders with default props', () => {
    render(<StatusCard />);
    const el = screen.getByTestId('status-card');
    expect(el).toBeDefined();
    expect(el.textContent).toContain('Status');
    expect(el.textContent).toContain('ok');
  });
});
