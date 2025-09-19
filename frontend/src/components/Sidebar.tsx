import React, { useState } from "react";
import { Link } from "react-router-dom";

type SidebarProps = {
  initial?: string;
};

const Sidebar: React.FC<SidebarProps> = ({ initial = "dashboard" }) => {
  const [active, setActive] = useState<string>(initial);

  return (
    <div className="w-64 h-screen bg-gray-900 text-white p-4 flex flex-col">
      <h2 className="text-2xl font-bold mb-8">Quantum Trader</h2>
      <nav className="flex-1">
        <ul>
          <li
            className={`mb-4 p-2 rounded-lg ${
              active === "dashboard" ? "bg-gray-700" : ""
            }`}
          >
            <Link to="/dashboard" onClick={() => setActive("dashboard")}>
              Dashboard
            </Link>
          </li>
          <li
            className={`mb-4 p-2 rounded-lg ${
              active === "settings" ? "bg-gray-700" : ""
            }`}
          >
            <Link to="/settings" onClick={() => setActive("settings")}>
              Settings
            </Link>
          </li>
        </ul>
      </nav>
      <div className="text-sm text-gray-400 mt-8">
        Quantum Trader © {new Date().getFullYear()}
      </div>
    </div>
  );
};

export default Sidebar;
