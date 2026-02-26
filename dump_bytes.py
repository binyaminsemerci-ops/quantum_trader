d = open('/opt/quantum/microservices/ai_engine/service.py', 'rb').read()
idx = d.find(b'atr_value is None')
print(repr(d[idx-60:idx+1000]))
