from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

(function() {
    // Override API base URL to match your running backend
    window.API_BASE_URL = "http://127.0.0.1:8000";
    
    console.log("API URL configured to:", window.API_BASE_URL);
    
    // Override fetch for API calls
    const originalFetch = window.fetch;
    window.fetch = function(url, options) {
        // If it's a relative URL for API endpoints, use our base URL
        if (typeof url === 'string' && url.startsWith('/') && 
            !url.startsWith('/assets/') && !url.startsWith('/manifest.json')) {
            console.log("Redirecting API call to backend:", url);
            url = window.API_BASE_URL + url;
        }
        return originalFetch(url, options);
    };
})();