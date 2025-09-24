$file = 'api_bodies/run_17960814021_console.log'
$lines = Get-Content -LiteralPath $file
# Find where ruff reported changes
$idx = ($lines | Select-String -Pattern 'Files modified by ruff' -SimpleMatch | Select-Object -First 1).LineNumber - 1
if (-not $idx) { Write-Output 'Could not find ruff changes marker'; exit 0 }
$start = [Math]::Max(0, $idx - 10)
$end = [Math]::Min($lines.Count - 1, $idx + 40)
for ($i = $start; $i -le $end; $i++) { Write-Output ("{0,4}: {1}" -f ($i+1), $lines[$i]) }

Write-Output "\n--- Bottom of snippet ---\n"
# Find the allow flag
$allowLine = ($lines | Select-String -Pattern 'echo "allow=' -SimpleMatch | Select-Object -First 1)
if ($allowLine) { Write-Output "Found allow line: $($allowLine.Line.Trim())" } else { Write-Output 'No explicit allow echo line found in snippet search' }

# Check for git commit/push within the snippet range
$push = $lines[$start..$end] -match 'git push|git commit'
if ($push) { Write-Output 'Found git commit/push text in snippet range:'; ($lines[$start..$end] | Select-String -Pattern 'git push|git commit' -AllMatches).ForEach({Write-Output $_.Line.Trim()}) } else { Write-Output 'No git commit/push lines in the snippet range' }