export default function Header(): JSX.Element {
  return (
    <header className="bg-gray-800 text-white p-4 flex justify-between items-center shadow">
      <h1 className="text-xl font-bold">Quantum Trader Dashboard</h1>
      <div className="text-sm text-gray-300">Live Trading Monitor</div>
    </header>
  );
}
