try {  = Invoke-WebRequest -UseBasicParsing http://127.0.0.1:39999/;  FETCH39999_OK Status= Len=0 } catch { FETCH39999_ERR  }
