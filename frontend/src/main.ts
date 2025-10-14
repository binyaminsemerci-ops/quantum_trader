import './style.css'
import './components/Dashboard.css'

// AI Model status tracking
let modelStatus = 'Loading...';
let lastTrainingDate = null;
let modelAccuracy = null;

// Use correct backend URL
const API_BASE_URL = 'http://127.0.0.1:8000';

// Fetch AI model metadata
async function fetchModelStatus() {
  try {
    const response = await fetch(`${API_BASE_URL}/api/ai/model/status`);
    const data = await response.json();
    modelStatus = data.status || 'Ready';
    lastTrainingDate = data.training_date || null;
    modelAccuracy = data.accuracy || null;
    updateModelDisplay();
  } catch (error) {
    modelStatus = 'Offline';
    updateModelDisplay();
  }
}

// Update model status display
function updateModelDisplay() {
  const statusElement = document.querySelector('.ai-status');
  if (statusElement) {
    const statusColor = modelStatus === 'Ready' ? '#4caf50' : 
                       modelStatus === 'Loading...' ? '#ff9800' : '#f44336';
    
    statusElement.innerHTML = `
      <span style="color: ${statusColor}">● ${modelStatus}</span>
      ${lastTrainingDate ? `<br>Last trained: ${new Date(lastTrainingDate).toLocaleDateString()}` : ''}
      ${modelAccuracy ? `<br>Accuracy: ${(modelAccuracy * 100).toFixed(1)}%` : ''}
    `;
  }
}

// Fetch real-time signals from AI model
async function fetchAISignals() {
  try {
    const response = await fetch(`${API_BASE_URL}/api/ai/signals/latest`);
    const signals = await response.json();
    updateSignalsFeed(signals);
  } catch (error) {
    console.log('Using mock signals - backend not available');
    updateSignalsFeed([]);
  }
}

// Update signals display
function updateSignalsFeed(signals) {
  const signalsContainer = document.querySelector('.signals-list');
  if (!signalsContainer) return;

  if (signals.length === 0) {
    signalsContainer.innerHTML = '<p>Loading signals from AI model...</p>';
    return;
  }

  const signalsHtml = signals.map(signal => `
    <div class="signal-item signal-${signal.type.toLowerCase()}">
      <div class="signal-header">
        <span class="signal-type">${signal.type}</span>
        <span class="signal-confidence">${(signal.confidence * 100).toFixed(1)}%</span>
      </div>
      <div class="signal-details">
        <span class="symbol">${signal.symbol}</span>
        <span class="price">$${signal.price.toFixed(2)}</span>
        <span class="time">${new Date(signal.timestamp).toLocaleTimeString()}</span>
      </div>
      <div class="signal-reason">${signal.reason}</div>
    </div>
  `).join('');

  signalsContainer.innerHTML = signalsHtml;
}

// Enhanced price update with real market data
async function updatePrice() {
  try {
    const response = await fetch(`${API_BASE_URL}/api/prices/latest?symbol=BTCUSDT`);
    const data = await response.json();
    
    const priceElement = document.querySelector('.price-display h3');
    const changeElement = document.querySelector('.price-change');
    
    if (priceElement && data.price) {
      priceElement.textContent = `Latest: ${data.price.toFixed(2)}`;
      
      if (changeElement && data.change !== undefined) {
        const changeClass = data.change >= 0 ? 'positive' : 'negative';
        const changeSign = data.change >= 0 ? '+' : '';
        changeElement.className = `price-change ${changeClass}`;
        changeElement.textContent = `${changeSign}${data.change.toFixed(2)} (${changeSign}${data.change_percent.toFixed(2)}%)`;
      }
    }
  } catch (error) {
    // Fallback to mock data if backend not available
    const priceElement = document.querySelector('.price-display h3');
    if (priceElement) {
      const newPrice = (45000 + (Math.random() - 0.5) * 2000).toFixed(2);
      priceElement.textContent = `Latest: ${newPrice}`;
    }
  }
}

// Enhanced dashboard HTML with AI status
document.querySelector<HTMLDivElement>('#app')!.innerHTML = `
  <div class="dashboard-container">
    <div class="dashboard-header">
      <h1>Quantum Trading Dashboard</h1>
      <div class="ai-status-container">
        <h4>AI Model Status</h4>
        <div class="ai-status">Loading...</div>
      </div>
    </div>
    <div class="dashboard-grid">
      <div class="price-section">
        <h2>Price Chart</h2>
        <div class="price-display">
          <h3>Latest: 45000.00</h3>
          <div class="price-change">+0.00 (0.00%)</div>
        </div>
        <div class="chart-placeholder">📈 Live Binance Data Chart</div>
        <div class="model-training-info">
          <p><strong>Data Sources:</strong></p>
          <ul>
            <li>📊 Binance Klines API (OHLCV)</li>
            <li>💭 CoinGecko Sentiment Analysis</li>
            <li>🤖 XGBoost Classification Model</li>
          </ul>
        </div>
      </div>
      <div class="signals-section">
        <h3>AI Trading Signals</h3>
        <div class="page-size-selector">
          <label>Page size: </label>
          <select id="page-size">
            <option value="10">10</option>
            <option value="25">25</option>
            <option value="50">50</option>
          </select>
          <button id="retrain-btn" class="retrain-button">🔄 Retrain Model</button>
        </div>
        <div class="signals-list">
          <p>Loading AI signals...</p>
        </div>
      </div>
    </div>
  </div>
`;

// Manual retrain trigger
document.getElementById('retrain-btn')?.addEventListener('click', async () => {
  const button = document.getElementById('retrain-btn') as HTMLButtonElement;
  if (button) {
    button.disabled = true;
    button.textContent = '🔄 Training...';
    
    try {
      const response = await fetch(`${API_BASE_URL}/api/ai/retrain`, { method: 'POST' });
      const result = await response.json();
      
      if (result.status === 'success') {
        alert('Model retraining started successfully!');
      } else {
        alert('Error starting retraining: ' + result.message);
      }
      
      setTimeout(() => {
        button.disabled = false;
        button.textContent = '🔄 Retrain Model';
        fetchModelStatus();
      }, 3000);
    } catch (error) {
      alert('Error communicating with backend');
      button.disabled = false;
      button.textContent = '🔄 Retrain Model';
    }
  }
});

// Initialize dashboard
async function initializeDashboard() {
  await fetchModelStatus();
  await fetchAISignals();
  updatePrice();
  
  // Update data every 10 seconds
  setInterval(updatePrice, 10000);
  setInterval(fetchAISignals, 30000);
  setInterval(fetchModelStatus, 60000);
}

// Start the dashboard
initializeDashboard();
