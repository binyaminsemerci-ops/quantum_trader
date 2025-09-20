// @ts-nocheck
// Consolidated safe fetch wrapper
(function() {
  const looksLikeApiPath = (u) => typeof u === 'string' && (u.startsWith('/api') || u.startsWith('/v1') || u.includes('/sentiment'));
  const originalFetch = window.fetch.bind(window);
  window.fetch = async function(url, options) {
    try {
      let finalUrl = url;
      if (looksLikeApiPath(typeof url === 'string' ? url : (url && url.url))) {
        const base = (window as any).API_BASE_URL || '';
        if (typeof url === 'string' && base) finalUrl = base + url;
      }
      const res = await originalFetch(finalUrl, options);
      if ((typeof finalUrl === 'string') && finalUrl.includes('sentiment')) {
        try {
          const clone = res.clone();
          const data = await clone.json();
          const fixed = {...data};
          fixed.positive_count = data.positive_count ?? data.positive ?? 0;
          fixed.negative_count = data.negative_count ?? data.negative ?? 0;
          fixed.neutral_count = data.neutral_count ?? data.neutral ?? 0;
          fixed.total_count = fixed.positive_count + fixed.negative_count + fixed.neutral_count;
          return new Response(JSON.stringify(fixed), { status: res.status, statusText: res.statusText, headers: res.headers });
        } catch (e) {
          return res;
        }
      }
      return res;
    } catch (e) {
      return originalFetch(url, options);
    }
  };
})();
