import { useEffect, useState } from 'react';

interface WatchItem { symbol: string; price: number; changePct: number }

interface WatchlistWidgetProps { symbols?: string[] }

export default function WatchlistWidget({ symbols }: WatchlistWidgetProps) {
  const [items, setItems] = useState<WatchItem[]>([]);
  const [newSymbol, setNewSymbol] = useState('');
  const [loading, setLoading] = useState(true);

  // Fetch watchlist from backend
  useEffect(() => {
    let cancelled = false;
    
    async function fetchWatchlist() {
      try {
        const response = await fetch('/watchlist');
        if (!response.ok) throw new Error('Failed to fetch watchlist');
        const watchlistData = await response.json();
        
        if (cancelled) return;
        
        // Get recent prices for watchlist symbols
        const pricePromises = (watchlistData || []).map(async (entry: any) => {
          try {
            const priceResponse = await fetch(`/prices/recent?symbol=${entry.symbol}&limit=2`);
            const priceData = await priceResponse.json();
            
            let price = 100; // fallback
            let changePct = 0;
            
            if (Array.isArray(priceData) && priceData.length > 0) {
              price = priceData[0].close || priceData[0].price || 100;
              
              // Calculate change if we have multiple data points
              if (priceData.length > 1) {
                const current = priceData[0].close || priceData[0].price;
                const previous = priceData[1].close || priceData[1].price;
                changePct = ((current - previous) / previous) * 100;
              }
            }
            
            return {
              symbol: entry.symbol,
              price: price,
              changePct: changePct
            };
          } catch {
            return {
              symbol: entry.symbol,
              price: 100,
              changePct: 0
            };
          }
        });
        
        const watchItems = await Promise.all(pricePromises);
        setItems(watchItems);
        setLoading(false);
      } catch (error) {
        console.error('Failed to fetch watchlist:', error);
        if (!cancelled) {
          // Fallback to showing active trading symbols from portfolio
          try {
            const portfolioResponse = await fetch('/portfolio/portfolio');
            const portfolioData = await portfolioResponse.json();
            const fallbackItems = (portfolioData.positions || []).map((pos: any) => ({
              symbol: pos.symbol,
              price: pos.current_price || 100,
              changePct: pos.pnlPercent || 0
            }));
            setItems(fallbackItems);
          } catch {
            setItems([]);
          }
          setLoading(false);
        }
      }
    }
    
    fetchWatchlist();
    
    return () => {
      cancelled = true;
    };
  }, [symbols]);

  // Update prices periodically
  useEffect(() => {
    if (items.length === 0) return;
    
    const id = setInterval(async () => {
      const updatedItems = await Promise.all(items.map(async (item) => {
        try {
          const response = await fetch(`/prices/recent?symbol=${item.symbol}&limit=2`);
          const priceData = await response.json();
          
          if (Array.isArray(priceData) && priceData.length > 0) {
            const price = priceData[0].close || priceData[0].price || item.price;
            let changePct = item.changePct;
            
            if (priceData.length > 1) {
              const current = priceData[0].close || priceData[0].price;
              const previous = priceData[1].close || priceData[1].price;
              changePct = ((current - previous) / previous) * 100;
            }
            
            return { ...item, price, changePct };
          }
        } catch {
          // Keep existing values on error
        }
        return item;
      }));
      
      setItems(updatedItems);
    }, 10000); // Update every 10 seconds
    
    return () => clearInterval(id);
  }, [items]);

  const add = async () => {
    const sym = newSymbol.trim().toUpperCase();
    if (!sym || items.find(i => i.symbol === sym)) return;
    
    try {
      // Add to backend watchlist
      await fetch('/watchlist', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ symbol: sym })
      });
      
      // Get current price for the new symbol
      let price = 100;
      let changePct = 0;
      
      try {
        const priceResponse = await fetch(`/prices/recent?symbol=${sym}&limit=2`);
        const priceData = await priceResponse.json();
        
        if (Array.isArray(priceData) && priceData.length > 0) {
          price = priceData[0].close || priceData[0].price || 100;
          
          if (priceData.length > 1) {
            const current = priceData[0].close || priceData[0].price;
            const previous = priceData[1].close || priceData[1].price;
            changePct = ((current - previous) / previous) * 100;
          }
        }
      } catch {
        // Use fallback values
      }
      
      setItems([...items, { symbol: sym, price, changePct }]);
      setNewSymbol('');
    } catch (error) {
      console.error('Failed to add symbol to watchlist:', error);
    }
  };

  return (
    <div className="h-full flex flex-col space-y-3">
      <div className="flex items-center space-x-2 text-xs">
        <input
          className="flex-1 px-2 py-1 rounded border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 focus:outline-none focus:ring focus:ring-indigo-500/30"
          placeholder="Legg til symbol"
          value={newSymbol}
          onChange={e => setNewSymbol(e.target.value)}
          onKeyDown={e => { if (e.key === 'Enter') add(); }}
        />
        <button onClick={add} className="px-2 py-1 rounded bg-indigo-600 text-white hover:bg-indigo-700">+</button>
      </div>
      <div className="flex-1 overflow-auto space-y-1 pr-1">
        {loading ? (
          <div className="flex items-center justify-center py-4">
            <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-500"></div>
          </div>
        ) : items.length === 0 ? (
          <div className="text-center py-4 text-gray-500 text-xs">
            Ingen symboler i watchlist
          </div>
        ) : (
          items.map(it => {
            const pos = it.changePct >= 0;
            return (
              <div key={it.symbol} className="flex items-center justify-between text-xs font-mono bg-white dark:bg-gray-700 rounded px-2 py-1 border border-gray-200 dark:border-gray-600">
                <span className="font-semibold truncate w-20">{it.symbol.replace('USDC','/USDC')}</span>
                <span className="text-gray-600 dark:text-gray-300">{it.price.toLocaleString()}</span>
                <span className={pos ? 'text-green-600' : 'text-red-500'}>{pos?'+':''}{it.changePct.toFixed(2)}%</span>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}
