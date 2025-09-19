import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  TimeScale
} from 'chart.js';
import 'chartjs-adapter-moment';
import moment from 'moment';
import { Signal, OHLCV } from '../../types';
import type { ChartData, ChartOptions } from 'chart.js';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend, TimeScale);

type PriceChartProps = {
  symbol?: string;
  interval?: string;
};

// Local aliases use shared types

const PriceChart: React.FC<PriceChartProps> = ({ symbol = 'BTCUSDT', interval = '1h' }) => {
  const [chartData, setChartData] = useState<ChartData<'line'> | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [signals, setSignals] = useState<Signal[]>([]);

  useEffect(() => {
    let mounted = true;

    const fetchData = async () => {
      setLoading(true);
      setError(null);
      try {
        const encodedSymbol = encodeURIComponent(String(symbol ?? ''));
        const encodedInterval = encodeURIComponent(String(interval ?? '1h'));
        const response = await axios.get<OHLCV[] | null>(`http://localhost:8000/ohlcv/${encodedSymbol}/${encodedInterval}`);
        const signalsResponse = await axios.get<Signal[] | null>(`http://localhost:8000/signals?symbol=${encodedSymbol}&limit=100`);

        const data: OHLCV[] = Array.isArray(response?.data) ? response!.data as OHLCV[] : [];
        const signalsData: Signal[] = Array.isArray(signalsResponse?.data) ? signalsResponse!.data as Signal[] : [];

        const labels: string[] = data.map((item: OHLCV) => (item && item.timestamp ? moment(item.timestamp).format('YYYY-MM-DD HH:mm') : ''));

        const pricesRaw: Array<number | null> = data.map((item: OHLCV) => {
          const raw = (item && (item.close ?? item.c ?? item.price ?? item.close)) ?? null;
          const n = raw === null ? NaN : Number(raw);
          return Number.isFinite(n) ? n : null;
        });

        // Chart.js expects numeric or null values; avoid passing NaN which breaks rendering
        const prices: number[] = pricesRaw.map((v) => (v === null ? NaN : v)) as number[];

        const computedChartData: ChartData<'line'> = {
          labels,
          datasets: [
            {
              label: `${symbol} Price`,
              data: prices,
              borderColor: 'rgb(75, 192, 192)',
              tension: 0.1,
              fill: false
            }
          ]
        };

        if (!mounted) return;
        setChartData(computedChartData);
        setSignals(Array.isArray(signalsData) ? signalsData : []);
        setLoading(false);
      } catch (err: any) {
        if (!mounted) return;
        console.error('Error fetching price data:', err?.message ?? err);
        setError('Failed to load price data. Please try again later.');
        setLoading(false);
      }
    };

    fetchData();
    const timer = setInterval(fetchData, 60000);
    return () => {
      clearInterval(timer);
      mounted = false;
    };
  }, [symbol, interval]);

    // calculate safe time unit
    const timeUnit = (() => {
      try {
        return interval && typeof interval === 'string' && interval.includes('d') ? 'day' : 'hour';
      } catch {
        return 'hour';
      }
    })();

    const chartOptions: ChartOptions<'line'> = {
    responsive: true,
    maintainAspectRatio: false,
    scales: {
      x: {
        type: 'time' as const,
        time: {
          unit: timeUnit,
          tooltipFormat: 'YYYY-MM-DD HH:mm',
          displayFormats: {
            hour: 'MMM D, HH:mm',
            day: 'MMM D'
          }
        },
        title: { display: true, text: 'Date/Time' }
      },
      y: { title: { display: true, text: 'Price (USD)' } }
    },
    plugins: {
      tooltip: { mode: 'index', intersect: false },
      legend: { position: 'top' as const },
      title: { display: true, text: `${symbol} Price Chart (${interval})` }
    }
  };

  if (loading) return <div className="chart-container loading">Loading price data...</div>;
  if (error) return <div className="chart-container error">{error}</div>;
  if (!chartData) return <div className="chart-container">No data available</div>;

  return (
    <div className="chart-container" style={{ height: '400px' }}>
      <Line data={chartData} options={chartOptions} />
      {signals && signals.length > 0 && (
        <div className="signals-overlay">
          <h4>Recent Signals</h4>
          <ul>
            {(signals || []).slice(0, 5).map((signal: Signal, idx: number) => {
              const rawSig = signal && (signal.signal ?? signal.signal_type ?? (signal as any).signalType ?? (signal as any).type);
              const sig = rawSig ? String(rawSig) : '';
              const cls = `signal-item ${sig ? sig.toLowerCase() : 'unknown'}`.trim();

              const rawConf = signal && ((signal as any).confidence ?? (signal as any).confidence_score ?? (signal as any).confidencePercent ?? null);
              const confNumber = typeof rawConf === 'number' && Number.isFinite(rawConf) ? rawConf : null;

              const confDisplay = confNumber === null ? '—' : `${Math.round(confNumber * 100)}%`;

              const when = signal && signal.timestamp ? moment(signal.timestamp).format('YYYY-MM-DD HH:mm') : 'N/A';
              const keyId = signal?.id ?? (signal as any)?._id ?? `${symbol}-sig-${idx}`;

              return (
                <li key={String(keyId)} className={cls}>
                  {when}: <strong>{sig}</strong> (Confidence: {confDisplay})
                </li>
              );
            })}
          </ul>
        </div>
      )}
    </div>
  );
};

export default PriceChart;

