const http = require('http');
const PORT = process.env.TEST_PORT ? parseInt(process.env.TEST_PORT,10) : 40111;
const started = Date.now();
const srv = http.createServer((req,res)=>{
  res.writeHead(200, {'Content-Type':'text/plain'});
  res.end('OK_SIMPLE ' + (Date.now()-started));
});

srv.on('error', (e)=>{
  console.error('SERVER_ERROR', e.code, e.message);
});

srv.listen(PORT,'127.0.0.1', () => {
  console.log('SIMPLE_LISTEN', PORT);
});
