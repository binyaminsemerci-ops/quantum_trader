// This patch fixes the common issues in the Quantum Trader frontend
(function() {
    // Fix variable hoisting issue
    window.f = window.f || {};
    
    // Fix sentiment data structure
    const originalFetch = window.fetch;
    window.fetch = async function(...args) {
        const response = await originalFetch.apply(this, args);
        
        try {
            // Only process JSON responses from sentiment endpoints
            const url = args[0];
            if (typeof url === 'string' && url.includes('sentiment')) {
                const clone = response.clone();
                const data = await clone.json();
                
                // Ensure positive_count exists
                if (data && !data.positive_count && (data.positive || data.data?.positive || data.sentiments?.positive)) {
                    const positive = data.positive || data.data?.positive || data.sentiments?.positive || 0;
                    const negative = data.negative || data.data?.negative || data.sentiments?.negative || 0;
                    const neutral = data.neutral || data.data?.neutral || data.sentiments?.neutral || 0;
                    
                    data.positive_count = positive;
                    data.negative_count = negative;
                    data.neutral_count = neutral;
                    data.total_count = positive + negative + neutral;
                    
                    // Return modified response
                    return new Response(JSON.stringify(data), {
                        status: response.status,
                        statusText: response.statusText,
                        headers: response.headers
                    });
                }
            }
        } catch (e) {
            console.log('Error in fetch override:', e);
        }
        
        return response;
    };
})();