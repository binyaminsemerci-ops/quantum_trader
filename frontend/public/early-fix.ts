// @ts-nocheck
window.onerror = function(msg, src, line, col, error) {
  console.error('Global Error:', msg, src, line);
  if (document.body && (!document.body.children.length || document.body.innerHTML === '')) {
    document.body.innerHTML = '<div style="padding:20px;font-family:sans-serif;">' +
      '<h1>Quantum Trader</h1><p>Noe gikk galt under lasting. Prøver å fikse...</p>' +
      '<button onclick="location.reload()">Last inn på nytt</button></div>';
  }
  return true;
};
window.f = {};
