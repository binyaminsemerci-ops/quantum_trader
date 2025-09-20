(function() {
    // Feilhåndtering for hele siden
    window.addEventListener('error', function(event) {
        console.error('Globalt JavaScript-feil:', event.error);
        // Forhindre hvit skjerm ved å vise en feilmelding
        if (document.body && document.body.innerHTML === '') {
            document.body.innerHTML = '<div style="padding:20px;font-family:sans-serif;"><h1>Quantum Trader</h1><p>Det oppstod en feil. Prøver å gjenopprette...</p><button onclick="location.reload()">Last inn på nytt</button></div>';
        }
        return false;
    });

    // Forbedret variabel initialisering
    window.f = window.f || {};
    
    // Initialisere alle mulige manglende variabler
    window.sentimentData = window.sentimentData || {};
    window.chartData = window.chartData || {};
    window.signalData = window.signalData || [];
    
    // Overvåk API-kall og legg til automatisk retry
    const originalFetch = window.fetch;
    window.fetch = async function(url, options) {
        console.log('API-kall:', url);
        
        // Redirect lokale API-kall til backend
        if (typeof url === 'string' && url.startsWith('/') && 
            !url.startsWith('/assets/') && !url.startsWith('/manifest.json')) {
            const newUrl = 'http://127.0.0.1:8000' + url;
            console.log('Omdirigerer til backend:', newUrl);
            url = newUrl;
        }
        
        // Prøv API-kall med retry
        let retries = 3;
        while (retries > 0) {
            try {
                const response = await originalFetch(url, options);
                
                // Håndter sentiment-data for å fikse formatet
                if (url.includes('sentiment')) {
                    const clone = response.clone();
                    try {
                        const data = await clone.json();
                        
                        // Sikre at alle nødvendige felt eksisterer
                        if (data) {
                            // Kopierer data før modifikasjon
                            const fixedData = {...data};
                            
                            fixedData.positive_count = data.positive_count || data.positive || 50;
                            fixedData.negative_count = data.negative_count || data.negative || 20;
                            fixedData.neutral_count = data.neutral_count || data.neutral || 30;
                            fixedData.total_count = fixedData.positive_count + fixedData.negative_count + fixedData.neutral_count;
                            
                            console.log('Fikset sentiment-data:', fixedData);
                            
                            return new Response(JSON.stringify(fixedData), {
                                status: response.status,
                                statusText: response.statusText,
                                headers: response.headers
                            });
                        }
                    } catch (e) {
                        console.error('Feil ved parsing av JSON:', e);
                    }
                }
                
                return response;
            } catch (error) {
                retries--;
                console.error(`API-kall feilet. ${retries} forsøk igjen:`, error);
                if (retries === 0) throw error;
                await new Promise(r => setTimeout(r, 1000)); // Vent 1 sekund før retry
            }
        }
    };

    // Vent på DOM-lasting
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initApp);
    } else {
        initApp();
    }

    // Initialiser app-funksjoner
    function initApp() {
        console.log('App initialiseres');
        
        // Forhindre caching
        const meta = document.createElement('meta');
        meta.httpEquiv = 'Cache-Control';
        meta.content = 'no-cache, no-store, must-revalidate';
        document.head.appendChild(meta);
        
        // Legg til en statusindikatorer
        setTimeout(function() {
            if (document.body) {
                const statusDiv = document.createElement('div');
                statusDiv.id = 'app-status';
                statusDiv.style.position = 'fixed';
                statusDiv.style.bottom = '10px';
                statusDiv.style.right = '10px';
                statusDiv.style.background = 'rgba(0,0,0,0.7)';
                statusDiv.style.color = 'white';
                statusDiv.style.padding = '5px 10px';
                statusDiv.style.borderRadius = '5px';
                statusDiv.style.fontSize = '12px';
                statusDiv.style.zIndex = '9999';
                statusDiv.textContent = 'Fiks aktivert';
                document.body.appendChild(statusDiv);
            }
        }, 1000);
    }
})();