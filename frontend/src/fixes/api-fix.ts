// Consolidated safe API fetch wrapper for Quantum Trader (TypeScript)
(function(){
  if ((window as any).__quantum_api_fix_installed) return;
  (window as any).__quantum_api_fix_installed = true;

  const API_BASE = typeof (window as any).API_BASE_URL === 'string' && (window as any).API_BASE_URL.length > 0
    ? (window as any).API_BASE_URL
    : null;

  const originalFetch = window.fetch.bind(window);

  const apiPathMatchers: RegExp[] = [
    /^\/api\b/i,
    /^\/sentiment\b/i,
    /^\/v1\//i
  ];

  function looksLikeApiPath(url: string | URL): boolean{
    if (typeof url === 'string'){
      try{ const u = new URL(url, window.location.href); return apiPathMatchers.some(m=>m.test(u.pathname)); }
      catch{ return apiPathMatchers.some(m=>m.test(url)); }
    }
    return apiPathMatchers.some(m=>m.test((url as URL).pathname));
  }

  async function normalizeSentimentResponse(response: Response){
    try{
      if (!response || !response.clone) return response;
      const clone = response.clone();
      const contentType = clone.headers.get('content-type') || '';
      if (!contentType.includes('application/json')) return response;
      const data = await clone.json();
      if (!data) return response;
      if (!/sentiment/i.test(response.url)) return response;

      const positive = data.positive_count ?? data.positive ?? data.data?.positive ?? 0;
      const negative = data.negative_count ?? data.negative ?? data.data?.negative ?? 0;
      const neutral = data.neutral_count ?? data.neutral ?? data.data?.neutral ?? 0;
      const fixed = Object.assign({}, data, {
        positive_count: positive,
        negative_count: negative,
        neutral_count: neutral,
        total_count: (positive||0) + (negative||0) + (neutral||0)
      });

      const headers: Record<string,string> = {};
      clone.headers.forEach((v,k)=>headers[k]=v);
      return new Response(JSON.stringify(fixed), { status: response.status, statusText: response.statusText, headers });
    }catch(e){
      console.warn('api-fix: failed to normalize sentiment response', e);
      return response;
    }
  }

  // assign as any to avoid strict function type mismatch with DOM lib types
  (window.fetch as any) = async function(input: RequestInfo | URL, init?: RequestInit){
    try{
      const url = typeof input === 'string' ? input : (input as Request).url || '';
      if (looksLikeApiPath(url)){
        const resolved = API_BASE ? (API_BASE.replace(/\/$/, '') + (url.toString().startsWith('/')?url.toString():('/'+url.toString()))) : url;
        const resp = await originalFetch(resolved, init);
        if (/sentiment/i.test(resolved.toString())) return normalizeSentimentResponse(resp);
        return resp;
      }
    }catch(e){
      console.error('api-fix fetch wrapper error', e);
    }
    return originalFetch(input, init);
  };

  try{
    if (typeof document !== 'undefined' && document.readyState !== 'loading'){
      const el = document.createElement('meta');
      el.name = 'quantum-api-fix';
      el.content = 'installed';
      document.head && document.head.appendChild(el);
    }
  }catch(e){}

})();
