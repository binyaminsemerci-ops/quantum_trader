// Minimal test HTTP server
const http = require('http');
const PORT = process.env.PORT || 6001;
const server = http.createServer((req,res)=>{
  res.writeHead(200, {'Content-Type':'text/plain'});
  res.end('QUICK_SERVER_OK ' + new Date().toISOString());
});
server.listen(PORT, '127.0.0.1', ()=>{
  console.log('[quick_server] listening on http://127.0.0.1:' + PORT);
});
