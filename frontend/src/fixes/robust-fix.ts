// Minimal safe shim to provide required globals (TypeScript)
(function(){
  if ((window as any).__quantum_robust_shim_installed) return;
  (window as any).__quantum_robust_shim_installed = true;
  (window as any).f = (window as any).f || {};
  (window as any).sentimentData = (window as any).sentimentData || {};
  (window as any).chartData = (window as any).chartData || {};
  (window as any).signalData = (window as any).signalData || [];
})();
