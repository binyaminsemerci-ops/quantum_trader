$path = 'api_bodies/run_17960814021_console.log'
$patterns = @('ruff','git commit','git push')
foreach ($p in $patterns) {
  Write-Host "\n=== PATTERN: $p ===\n"
  Select-String -Path $path -Pattern $p -SimpleMatch -Context 3 | ForEach-Object {
    Write-Host "---- Match at line $($_.LineNumber) ----"
    if ($_.Context.PreContext) { $_.Context.PreContext | ForEach-Object { Write-Host $_ } }
    Write-Host $_.Line
    if ($_.Context.PostContext) { $_.Context.PostContext | ForEach-Object { Write-Host $_ } }
  }
}
Write-Host '\nDone'
