param([int]$Port=40112)
$ErrorActionPreference='Stop'
$client = New-Object System.Net.Sockets.TcpClient
try {
  $client.Connect('127.0.0.1',$Port)
  if ($client.Connected) {"DOTNET_CLIENT_CONNECTED True"} else {"DOTNET_CLIENT_CONNECTED False"}
} catch {
  "DOTNET_CLIENT_ERR " + $_.Exception.GetType().Name + ": " + $_.Exception.Message
  exit 2
}
$client.Close()
