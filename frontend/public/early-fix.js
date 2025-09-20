// Denne koden må kjøre før alt annet
window.onerror = function(msg, src, line, col, error) {
    console.error("Global Error:", msg, "at", src, ":", line);
    // Vis alltid innhold i stedet for hvit skjerm
    if (document.body && (!document.body.children.length || document.body.innerHTML === '')) {
        document.body.innerHTML = '<div style="padding:20px;font-family:sans-serif;">' +
            '<h1>Quantum Trader</h1>' +
            '<p>Noe gikk galt under lasting. Prøver å fikse...</p>' +
            '<div id="error-details" style="background:#f8f8f8;padding:10px;margin:10px 0;font-family:monospace;"></div>' +
            '<button onclick="location.reload()" style="padding:10px;background:#4CAF50;color:white;border:none;cursor:pointer;">Last inn på nytt</button>' +
            '</div>';
        document.getElementById('error-details').textContent = msg + " at " + src + ":" + line;
    }
    return true; // La nettleseren også vise feilen i konsollet
};

// Forsikre oss om at grunnleggende variabler eksisterer
window.f = {};