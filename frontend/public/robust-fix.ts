// @ts-nocheck
(function() {
  window.f = window.f || {};
  window.sentimentData = window.sentimentData || {};
  window.chartData = window.chartData || {};
  window.signalData = window.signalData || [];

  // Minimal safe fetch wrapper (avoid hard redirects)
  const originalFetch = window.fetch.bind(window);
  window.fetch = async function(url, options) {
    try { return await originalFetch(url, options); } catch (e) { console.error('fetch error', e); throw e; }
  };
})();
