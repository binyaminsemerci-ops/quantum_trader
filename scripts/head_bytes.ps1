param([string]$Path)
$fs = [System.IO.File]::OpenRead($Path)
$buf = New-Object byte[] 8
$read = $fs.Read($buf, 0, 8)
$fs.Close()
Write-Host ($buf[0..($read-1)] -join ' ')
