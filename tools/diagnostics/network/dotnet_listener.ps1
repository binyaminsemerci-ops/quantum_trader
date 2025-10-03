$ErrorActionPreference='Stop'
$Port = 40112
$l = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, $Port)
$l.Start()
"DOTNET_LISTEN $Port"
while ($true) {
  if ($l.Pending()) {
    $c = $l.AcceptTcpClient()
    "DOTNET_ACCEPT from " + $c.Client.RemoteEndPoint.ToString()
    $c.Close()
  }
  Start-Sleep -Milliseconds 250
}
