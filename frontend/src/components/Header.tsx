import React from "react";

export type HeaderProps = {
  title?: React.ReactNode;
  subtitle?: React.ReactNode;
};

const Header: React.FC<HeaderProps> = ({ title = 'Quantum Trader Dashboard', subtitle = 'Live Trading Monitor' }) => {
  return (
    <header className="bg-gray-800 text-white p-4 flex justify-between items-center shadow">
      <h1 className="text-xl font-bold">{title}</h1>
      <div className="text-sm text-gray-300">{subtitle}</div>
    </header>
  );
};

export default Header;
