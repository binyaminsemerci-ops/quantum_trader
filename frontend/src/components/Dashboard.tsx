import { useEffect, useState } from "react";
import StatsCard from "./StatsCard";
import ChartView from "./ChartView";
import TradeTable from "./TradeTable";
import RiskMonitor from "./RiskMonitor";
import LoaderOverlay from "./LoaderOverlay";

export default function Dashboard() {
	const [stats, setStats] = useState<any>(null);
		// chartData removed: ChartView consumes its own data via api.getChart()
	const [trades, setTrades] = useState<any[]>([]);
	const [loading, setLoading] = useState(true);

	useEffect(() => {
		// Dummy data for testing
		const dummyStats = {
			balance: 10000,
			equity: 10250,
			risk_percent: 1.5,
			open_trades: 3,
		};

		const dummyTrades = [
			{ date: "2025-09-01", pair: "BTC/USDT", side: "BUY", amount: 0.5, price: 25000 },
			{ date: "2025-09-02", pair: "ETH/USDT", side: "SELL", amount: 2, price: 1600 },
			{ date: "2025-09-03", pair: "BNB/USDT", side: "BUY", amount: 5, price: 280 },
		];

			setTimeout(() => {
				setStats(dummyStats);
				setTrades(dummyTrades);
				setLoading(false);
			}, 1000);
	}, []);

	if (loading) return <LoaderOverlay message="Loading dashboard..." />;

	return (
		<div className="p-6 space-y-6">
			{/* Stats */}
			<div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
				<StatsCard title="Balance" value={`$${stats?.balance ?? 0}`} />
				<StatsCard title="Equity" value={`$${stats?.equity ?? 0}`} />
				<StatsCard title="Risk %" value={`${stats?.risk_percent ?? 0}%`} />
				<StatsCard title="Open Trades" value={stats?.open_trades ?? 0} />
			</div>

			{/* Chart + Risk Monitor */}
			<div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
						<div className="lg:col-span-2">
							<ChartView title="Performance" />
						</div>
				<RiskMonitor />
			</div>

			{/* Trade Table */}
			<div className="mt-6">
				<TradeTable trades={trades} />
			</div>
		</div>
	);
}
