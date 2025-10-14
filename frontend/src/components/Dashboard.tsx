import React, { useState, useEffect } from 'react';

interface PriceData {
  price: number;
  timestamp: string;
  change: number;
  changePercent: number;
}

export const Dashboard: React.FC = () => {
  const [priceData, setPriceData] = useState<PriceData>({
    price: 149.00,
    timestamp: new Date().toISOString(),
    change: 0,
    changePercent: 0
  });
  
  const [pageSize, setPageSize] = useState(10);

  useEffect(() => {
    // Mock price updates every 5 seconds
    const interval = setInterval(() => {
      const newPrice = 149.00 + (Math.random() - 0.5) * 10;
      const change = newPrice - priceData.price;
      const changePercent = (change / priceData.price) * 100;
      
      setPriceData({
        price: newPrice,
        timestamp: new Date().toISOString(),
        change,
        changePercent
      });
    }, 5000);

    return () => clearInterval(interval);
  }, [priceData.price]);

  return (
    <div className="dashboard-container">
      <div className="dashboard-header">
        <h1>Dashboard</h1>
      </div>
      
      <div className="dashboard-grid">
        <div className="price-section">
          <h2>Price chart</h2>
          <div className="price-display">
            <h3>Latest: {priceData.price.toFixed(2)}</h3>
            <div className={`price-change ${priceData.change >= 0 ? 'positive' : 'negative'}`}>
              {priceData.change >= 0 ? '+' : ''}{priceData.change.toFixed(2)} 
              ({priceData.changePercent >= 0 ? '+' : ''}{priceData.changePercent.toFixed(2)}%)
            </div>
          </div>
          <div className="chart-placeholder">
            📈 Chart will be displayed here
          </div>
        </div>
        
        <div className="signals-section">
          <h3>Signal Feed (mock)</h3>
          <div className="page-size-selector">
            <label>Page size: </label>
            <select 
              value={pageSize} 
              onChange={(e) => setPageSize(Number(e.target.value))}
            >
              <option value={10}>10</option>
              <option value={25}>25</option>
              <option value={50}>50</option>
            </select>
          </div>
          
          <div className="signals-list">
            <p>No signals yet</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;