$file = 'api_bodies/run_17960814021_console.log'
$lines = Get-Content $file
$start = 820
$end = 900
for ($i = $start; $i -le $end -and $i -lt $lines.Count; $i++) {
    Write-Output ("{0,6}: {1}" -f ($i+1), $lines[$i])
}