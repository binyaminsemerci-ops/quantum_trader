// Deep diagnostic server
const http = require('http');
const events = [];
function log(msg){
  const line = `[server_debug ${new Date().toISOString()}] ${msg}`;
  console.log(line);
  events.push(line);
}
process.on('exit',c=>log('process exit code='+c));
process.on('uncaughtException',e=>log('uncaughtException '+e.stack));
process.on('unhandledRejection',e=>log('unhandledRejection '+e));

const server = http.createServer((req,res)=>{
  if(req.url==='/events'){res.writeHead(200,{'Content-Type':'application/json'});return res.end(JSON.stringify(events,null,2));}
  res.writeHead(200,{'Content-Type':'text/plain'});res.end('SERVER_DEBUG_OK '+Date.now());
});
server.listen(0,'127.0.0.1',()=>{
  const addr = server.address();
  log('listening on '+addr.address+':'+addr.port);
  setInterval(()=>log('heartbeat'),4000).unref();
});