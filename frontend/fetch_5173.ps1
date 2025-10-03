try {  = Invoke-WebRequest -UseBasicParsing http://127.0.0.1:5173/;  FETCH5173_OK Status= Len=0 } catch { FETCH5173_ERR  }
