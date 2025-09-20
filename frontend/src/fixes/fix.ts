// Minimal shim (TypeScript)
(function(){
  if ((window as any).__quantum_fix_shim_installed) return;
  (window as any).__quantum_fix_shim_installed = true;
  (window as any).f = (window as any).f || {};
})();
