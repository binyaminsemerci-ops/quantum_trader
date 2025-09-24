# poll_github_labeler.ps1
param(
  [string] $owner = 'binyaminsemerci-ops',
  [string] $repo  = 'quantum_trader',
  [int]    $prNumber    = 25,
  [int]    $pollIntervalSec = 15,
  [int]    $maxChecks = 40,
  [switch] $DumpJson = $false,
  [string] $workflowBranch = '',
  [switch] $UseWorkflowIdFallback = $false,
  [string] $GitHubToken = '',
  [string] $SaveApiBodies = '',
  [switch] $AutoComment = $false
)

if (-not $GitHubToken) { $GitHubToken = $env:GITHUB_TOKEN }
if (-not $GitHubToken) {
  Write-Error "Please provide a GitHub token either via -GitHubToken or set environment variable GITHUB_TOKEN before running (PAT with repo scope)."
  exit 2
}

$headers = @{ Authorization = "Bearer $GitHubToken"; Accept = 'application/vnd.github+json' }

# Helper: human readable HTTP status message
function HttpStatusMessage($code) {
  switch ($code) {
    200 { return 'OK' }
    201 { return 'Created' }
    204 { return 'No Content' }
    400 { return 'Bad Request' }
    401 { return 'Unauthorized - token missing/invalid or insufficient scope' }
    403 { return 'Forbidden - token lacks permission or rate-limited' }
    404 { return 'Not Found - workflow may not be registered on default branch' }
    422 { return 'Unprocessable Entity - validation error' }
    default { return "HTTP $code" }
  }
}

# Ensure save dir exists when requested
if ($SaveApiBodies) {
  try { New-Item -ItemType Directory -Path $SaveApiBodies -Force | Out-Null } catch { Write-Warning "Could not create SaveApiBodies dir: $SaveApiBodies" }
}

function Get-Json($url) {
  try {
    $resp = Invoke-RestMethod -Uri $url -Headers $headers -Method Get
    $global:LastApiStatus = 200
    try { $global:LastApiBody = $resp | ConvertTo-Json -Depth 10 } catch { $global:LastApiBody = "$($resp.GetType().FullName)" }
    if ($DumpJson) {
      Write-Host "--- API JSON from: $url ---"
      try { $resp | ConvertTo-Json -Depth 10 | Write-Host } catch { Write-Host ("(unable to convert response to JSON: {0})" -f $_.Exception.Message) }
      Write-Host "--- end JSON ---"
    }
    if ($SaveApiBodies) {
      try {
        $safe = ($url -replace '[^a-zA-Z0-9]','_')
        $fname = Join-Path $SaveApiBodies ("api_$(Get-Date -Format yyyyMMdd_HHmmss)_$safe.json")
        $resp | ConvertTo-Json -Depth 10 | Out-File -Encoding UTF8 -FilePath $fname
        Write-Host "Saved API body to: $fname"
  } catch { Write-Warning ("Failed to save API body for {0}: {1}" -f $url, $_.Exception.Message) }
    }
    return $resp
  } catch {
    Write-Warning "API request failed: $url"
    # Try to run a fallback to capture HTTP status and body
    try {
      $wr = Invoke-WebRequest -Uri $url -Headers $headers -Method Get -ErrorAction Stop
      $global:LastApiStatus = $wr.StatusCode.Value
      $global:LastApiBody = $wr.Content
      Write-Warning ("Fallback status: {0} - {1}" -f $wr.StatusCode, (HttpStatusMessage $wr.StatusCode.Value))
      if ($wr.Content) { Write-Warning ("Fallback body length: {0}" -f $wr.Content.Length) }
      if ($SaveApiBodies) {
        try {
          $safe = ($url -replace '[^a-zA-Z0-9]','_')
          $fname = Join-Path $SaveApiBodies ("api_err_$(Get-Date -Format yyyyMMdd_HHmmss)_$safe.txt")
          $wr.Content | Out-File -Encoding UTF8 -FilePath $fname
          Write-Host "Saved API error body to: $fname"
  } catch { Write-Warning ("Failed to save API error body for {0}: {1}" -f $url, $_.Exception.Message) }
      }
    } catch {
      # If Invoke-WebRequest also fails, try to print the exception message
      $global:LastApiStatus = $null
      $global:LastApiBody = $_.Exception.Message
      Write-Warning ("Fallback request failed: {0}" -f $_.Exception.Message)
      if ($SaveApiBodies) {
        try {
          $safe = ($url -replace '[^a-zA-Z0-9]','_')
          $fname = Join-Path $SaveApiBodies ("api_err_ex_$(Get-Date -Format yyyyMMdd_HHmmss)_$safe.txt")
          $global:LastApiBody | Out-File -Encoding UTF8 -FilePath $fname
          Write-Host "Saved API exception body to: $fname"
  } catch { Write-Warning ("Failed to save API exception body for {0}: {1}" -f $url, $_.Exception.Message) }
      }
    }
    return $null
  }
}

# Helper to print run summary
function Print-Run($r) {
  "{0} | workflow:{1} | status:{2} | conclusion:{3} | id:{4}" -f $r.created_at, ($r.name), ($r.status), ($r.conclusion), ($r.id)
}

# Get PR meta

$prUrl = "https://api.github.com/repos/$owner/$repo/pulls/$prNumber"
$prData = Get-Json $prUrl
if (-not $prData) { Write-Error "Could not fetch PR $prNumber"; exit 3 }

Write-Host ("Monitoring PR {0}/{1}#{2}: '{3}' (head: {4})" -f $owner, $repo, $prNumber, $prData.title, $prData.head.ref)
$labelNames = '(none)'
if ($prData -and $prData.labels) {
  $names = $prData.labels | ForEach-Object { $_.name }
  if ($names) { $labelNames = [string]::Join(', ', $names) }
}
Write-Host ("Current labels: {0}" -f $labelNames)

# Enumerate repository workflows early to help debug filename vs id lookups
try {
  $workflowsListUrl = "https://api.github.com/repos/$owner/$repo/actions/workflows"
  Write-Host "Fetching repository workflows list for diagnostic mapping: $workflowsListUrl"
  $workflowsList = Get-Json $workflowsListUrl
  if ($workflowsList -and $workflowsList.workflows) {
    Write-Host "Repository workflows (id | name | path):"
    $workflowsList.workflows | ForEach-Object { Write-Host ("{0} | {1} | {2}" -f $_.id, $_.name, $_.path) }
  } else {
    Write-Host "No workflows returned from repository workflows list."
  }
} catch {
  Write-Warning "Failed to enumerate repository workflows for diagnostics: $($_.Exception.Message)"
}

# Check comments and labels now and poll
$check = 0
while ($check -lt $maxChecks) {
  $check++
  Write-Host "=== Poll #$check ($(Get-Date)) ==="

  # Check for /auto-ruff-fix comment
  $commentsUrl = "https://api.github.com/repos/$owner/$repo/issues/$prNumber/comments"
  $comments = Get-Json $commentsUrl
  # Use a different local variable name to avoid colliding with the [switch] parameter $AutoComment
  $foundAutoComment = $comments | Where-Object { $_.body -match '/auto-ruff-fix' } | Select-Object -First 1
  if ($foundAutoComment) {
    Write-Host "Found /auto-ruff-fix comment by $($foundAutoComment.user.login) at $($foundAutoComment.created_at)"
  } else {
    Write-Host "No /auto-ruff-fix comment found yet."
  }

  # Optionally post a test comment (guarded)
  if ($AutoComment) {
    Write-Host "AutoComment flag set: posting a test '/auto-ruff-fix' comment to PR $prNumber (this will be visible on GitHub)."
    $postBody = @{ body = '/auto-ruff-fix' } | ConvertTo-Json
    try {
      $postResp = Invoke-RestMethod -Uri $commentsUrl -Headers $headers -Method Post -Body $postBody -ContentType 'application/json'
      Write-Host "Posted test comment by: $($postResp.user.login) at $($postResp.created_at)"
      if ($SaveApiBodies) { $postResp | ConvertTo-Json -Depth 6 | Out-File (Join-Path $SaveApiBodies ("posted_comment_$(Get-Date -Format yyyyMMdd_HHmmss).json")) }
    } catch {
      Write-Warning ("Failed to post test comment: {0}" -f $_.Exception.Message)
    }
  }

  # Refresh labels
  $newPr = Get-Json $prUrl
  if ($newPr) { $prData = $newPr }
  $hasLabel = $false
  if ($prData -and $prData.labels) {
    $hasLabel = ($prData.labels | Where-Object { $_.name -eq 'auto-ruff-fix' }) -ne $null
  }
  Write-Host "PR has label 'auto-ruff-fix'? $hasLabel"

  # Proactively fetch repository workflows list every poll so we can decide whether
  # to query runs by filename or by numeric workflow id (helps when workflows are
  # registered/removed on the default branch during testing).
  $workflowsUrl = "https://api.github.com/repos/$owner/$repo/actions/workflows"
  $workflows = Get-Json $workflowsUrl
  if ($workflows -and $workflows.workflows) {
    Write-Host ("Repository workflows fetched: {0}" -f ($workflows.workflows.Count))
  } else {
    Write-Host "Repository workflows not available this poll."
  }

  # Get recent runs of the labeler workflow
  $workflowLabeler = "auto_label_on_comment.yml"
  # Prefer querying runs by numeric workflow id when we can find the workflow in the repo list.
  $foundWf = $null
  if ($workflows -and $workflows.workflows) {
    $foundWf = $workflows.workflows | Where-Object { $_.path -eq ".github/workflows/$workflowLabeler" -or $_.path -eq "/.github/workflows/$workflowLabeler" -or $_.name -eq $workflowLabeler } | Select-Object -First 1
  }
  if ($foundWf) {
    $wfId = $foundWf.id
    $labelerRunsUrl = "https://api.github.com/repos/$owner/$repo/actions/workflows/$wfId/runs?per_page=5"
  } else {
    # Fallback to filename-based runs query; this may 404 when the workflow isn't registered on default branch
    $labelerRunsUrl = "https://api.github.com/repos/$owner/$repo/actions/workflows/$workflowLabeler/runs?per_page=5"
  }
  if ($workflowBranch) { $labelerRunsUrl += "&branch=$workflowBranch" }
  Write-Host "Querying labeler workflow runs URL: $labelerRunsUrl"
  $labelerRuns = Get-Json $labelerRunsUrl
  if ($labelerRuns -and $labelerRuns.workflow_runs) {
    Write-Host ("Recent runs of {0}:" -f $workflowLabeler)
    $labelerRuns.workflow_runs | ForEach-Object { Print-Run $_ }
  } else {
    if ($global:LastApiStatus -eq 404) {
      Write-Warning ("Workflow runs endpoint returned 404 for {0}. Workflow may not be registered on the default branch." -f $workflowLabeler)
      # Always attempt fallback: enumerate repository workflows to find numeric id
      Write-Host "Attempting fallback: enumerate repository workflows to find numeric id..."
      $workflowsUrl = "https://api.github.com/repos/$owner/$repo/actions/workflows"
      $workflows = Get-Json $workflowsUrl
      if ($workflows -and $workflows.workflows) {
        # Try to find by path then by name
        $found = $workflows.workflows | Where-Object { $_.path -eq "/.github/workflows/$workflowLabeler" -or $_.name -eq $workflowLabeler } | Select-Object -First 1
        if ($found) {
          $wfId = $found.id
          Write-Host ("Found workflow id {0} for {1}; querying runs by id..." -f $wfId, $workflowLabeler)
          $labelerRunsByIdUrl = "https://api.github.com/repos/$owner/$repo/actions/workflows/$wfId/runs?per_page=5"
          if ($workflowBranch) { $labelerRunsByIdUrl += "&branch=$workflowBranch" }
          Write-Host "Querying by id URL: $labelerRunsByIdUrl"
          $labelerRuns = Get-Json $labelerRunsByIdUrl
          if ($labelerRuns -and $labelerRuns.workflow_runs) {
            Write-Host ("Recent runs of {0} (by id):" -f $workflowLabeler)
            $labelerRuns.workflow_runs | ForEach-Object { Print-Run $_ }
          } else {
            Write-Host ("No recent runs found for {0} (by id)." -f $workflowLabeler)
          }
        } else {
          Write-Warning "Could not find workflow in repository workflows list."
        }
      } else {
        Write-Warning "Could not enumerate workflows for fallback (API returned no workflows)."
      }
    } else {
      Write-Host ("No recent runs found for {0}" -f $workflowLabeler)
    }
  }

  # Get recent runs of the CI workflow
  $workflowCI = "ci.yml"
  $ciRunsUrl = "https://api.github.com/repos/$owner/$repo/actions/workflows/$workflowCI/runs?per_page=5"
  if ($workflowBranch) { $ciRunsUrl += "&branch=$workflowBranch" }
  Write-Host "Querying CI workflow runs URL: $ciRunsUrl"
  $ciRuns = Get-Json $ciRunsUrl
  if ($ciRuns -and $ciRuns.workflow_runs) {
    Write-Host ("Recent runs of {0}:" -f $workflowCI)
    $ciRuns.workflow_runs | ForEach-Object { Print-Run $_ }
  } else {
    if ($global:LastApiStatus -eq 404) {
      Write-Host "CI runs endpoint returned 404; attempting workflow-id fallback for CI workflow..."
      $workflowsUrl = "https://api.github.com/repos/$owner/$repo/actions/workflows"
      $workflows = Get-Json $workflowsUrl
      if ($workflows -and $workflows.workflows) {
        $foundCi = $workflows.workflows | Where-Object { $_.path -eq "/.github/workflows/$workflowCI" -or $_.name -eq $workflowCI } | Select-Object -First 1
        if ($foundCi) {
          $wfId = $foundCi.id
          $ciRunsByIdUrl = "https://api.github.com/repos/$owner/$repo/actions/workflows/$wfId/runs?per_page=5"
          if ($workflowBranch) { $ciRunsByIdUrl += "&branch=$workflowBranch" }
          Write-Host "Querying CI by id URL: $ciRunsByIdUrl"
          $ciRuns = Get-Json $ciRunsByIdUrl
          if ($ciRuns -and $ciRuns.workflow_runs) {
            Write-Host ("Recent runs of {0} (by id):" -f $workflowCI)
            $ciRuns.workflow_runs | ForEach-Object { Print-Run $_ }
          } else {
            Write-Host ("No recent runs found for {0} (by id)." -f $workflowCI)
          }
        } else {
          Write-Warning "Could not find CI workflow in repository workflows list for fallback."
        }
      } else {
        Write-Warning "Could not enumerate workflows for CI fallback (API returned no workflows)."
      }
    } else {
      Write-Host ("No recent runs found for {0}" -f $workflowCI)
    }
  }

  # Check PR commits to detect auto-ruff commit
  $commitsUrl = "https://api.github.com/repos/$owner/$repo/pulls/$prNumber/commits"
  $commits = Get-Json $commitsUrl
  $autoCommit = $commits | Where-Object { ($_.commit.message -match 'apply ruff fixes') -or ($_.author -and $_.author.login -eq 'github-actions') } | Select-Object -First 1
  if ($autoCommit) {
    Write-Host "Found candidate auto-commit: $($autoCommit.sha) - $($autoCommit.commit.message)"
    Write-Host "Author: $($autoCommit.author.login) | Committer: $($autoCommit.commit.committer.name)"
    break
  } else {
    Write-Host "No auto-commit detected in PR commits yet."
  }

  Start-Sleep -Seconds $pollIntervalSec
}

Write-Host "Polling finished."