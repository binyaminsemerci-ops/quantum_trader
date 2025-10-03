import http.server, socketserver, threading, urllib.request, time, sys
PORT=40333
class H(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a, **k):
        pass
try:
    httpd = socketserver.TCPServer(('127.0.0.1', PORT), H)
except Exception as e:
    print('PY_LISTEN_ERR', type(e).__name__, str(e))
    sys.exit(2)
print('PY_LISTEN', PORT)
threading.Thread(target=httpd.serve_forever, daemon=True).start()
ok=False
for i in range(10):
    time.sleep(0.4)
    try:
        with urllib.request.urlopen(f'http://127.0.0.1:{PORT}/') as r:
            print('PY_FETCH_OK', r.status)
            ok=True
            break
    except Exception as e:
        print('PY_FETCH_ERR', type(e).__name__, str(e))
print('PY_RESULT', ok)
httpd.shutdown()
