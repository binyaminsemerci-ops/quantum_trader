param()

Write-Host 'Recovering archived run logs from git history...'

$files = @(
  'run-18052868849-log.txt',
  'run-18052868980-log.txt',
  'run-18052868998-log.txt',
  'run-18052868999-log.txt',
  'run-18052869000-log.txt',
  'run-18052869006-log.txt',
  'run-18052869009-log.txt',
  'run-18053022341-log.txt',
  'run-18053068411-log.txt',
  'run-54-logs.txt',
  'run-18053570799-logs.zip'
)

New-Item -ItemType Directory -Force -Path "artifacts\archived-runs" | Out-Null

foreach ($f in $files) {
  try {
    $commit = git rev-list -n 1 --all -- $f 2>$null
  } catch {
    $commit = $null
  }
  if (![string]::IsNullOrEmpty($commit)) {
    Write-Host "Recovering $f from commit $commit"
    try {
      git show "$commit`:$f" | Out-File -Encoding utf8 "artifacts\archived-runs\$f"
    } catch {
      Write-Host ('Failed to extract ' + $f + ': ' + $_)
    }
  } else {
    Write-Host "Not found in history: $f"
  }
}

if (Test-Path 'artifacts\archived-runs') {
  Write-Host 'Creating ZIP archive: artifacts/archived-runs.zip'
  Compress-Archive -Path 'artifacts\archived-runs\*' -DestinationPath 'artifacts\archived-runs.zip' -Force
  Write-Host 'Archive created.'
} else {
  Write-Host 'No archived files found.'
}

Write-Host 'Done.'
