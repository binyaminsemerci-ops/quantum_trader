$token = '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'
$headers = @{ Authorization = "Bearer $token"; Accept = 'application/vnd.github+json' }
$owner = 'binyaminsemerci-ops'
$repo = 'quantum_trader'
$pr = 25

try {
  $issue = Invoke-RestMethod -Uri ("https://api.github.com/repos/$owner/$repo/issues/$pr") -Headers $headers -Method Get
} catch {
  Write-Error "Failed to fetch issue: $($_.Exception.Message)"; exit 2
}

$labels = ($issue.labels | ForEach-Object { $_.name }) -join ', '
if (-not $labels) { $labels = '(none)' }
Write-Host "Labels: $labels"

Write-Host "Latest comments (time | user | body):"
try {
  $comments = Invoke-RestMethod -Uri ("https://api.github.com/repos/$owner/$repo/issues/$pr/comments") -Headers $headers -Method Get
  $last = $comments | Select-Object -Last 10
  foreach ($c in $last) {
    $b = $c.body
    $b = $b -replace "`r`n", ' ↵ '
    $b = $b -replace "`n", ' ↵ '
    Write-Host ("{0} | {1} | {2}" -f $c.created_at, $c.user.login, $b)
  }
} catch {
  Write-Warning "Failed to fetch comments: $($_.Exception.Message)"
}
