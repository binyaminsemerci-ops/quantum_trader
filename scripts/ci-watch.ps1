<#
Poll GitHub Actions runs and download frontend audit artifacts for failed runs.

Usage:
  pwsh .\scripts\ci-watch.ps1 [-PollIntervalSeconds 60] [-MaxRuns 50] [-Once]

Requirements:
  - GitHub CLI (gh) must be installed and authenticated (gh auth login).
  - This script runs on Windows PowerShell / pwsh.

It will create folders under artifacts/ci-runs/<run-id>/ and save downloaded
artifacts there along with a small JSON metadata file.
#>
param(
    [int]$PollIntervalSeconds = 60,
    [int]$MaxRuns = 100,
    [switch]$Once
)

$Repo = "binyaminsemerci-ops/quantum_trader"
$OutDir = "artifacts/ci-runs"
if (-not (Test-Path $OutDir)) { New-Item -ItemType Directory -Path $OutDir | Out-Null }
$LogFile = Join-Path $OutDir "monitor.log"
Function Log([string]$msg) {
    $time = (Get-Date).ToString("s")
    $line = "$time`t$msg"
    Add-Content -Path $LogFile -Value $line
    Write-Host $line
}

Log "Starting CI watcher for repo $Repo"

# The credential target name the watcher/installer agree on. Logged for clarity.
$expectedCredTarget = 'QuantumTraderCIWatcher_GH'
Log "Auth: expected credential target name: $expectedCredTarget"

# AUTH: determine source of GitHub token and log clearly for verification
$authSource = $null
if ($env:GH_TOKEN) {
    $authSource = 'environment'
    Log "Auth: GH_TOKEN found in environment. (This may come from NSSM AppEnvironmentExtra or the current session.)"
} else {
    # Try to load from Windows Credential Manager (target: <ServiceName>_GH or QuantumTraderCIWatcher_GH)
    try {
        Import-Module CredentialManager -ErrorAction SilentlyContinue
    } catch {}

    if (Get-Module -ListAvailable -Name CredentialManager) {
        try {
            $stored = Get-StoredCredential -Target 'QuantumTraderCIWatcher_GH' -ErrorAction SilentlyContinue
            if (-not $stored) {
                # also try the ServiceName-based target as a secondary location
                $stored = Get-StoredCredential -Target 'QuantumTraderCIWatcher_GH' -ErrorAction SilentlyContinue
            }
            if ($stored -and $stored.Password) {
                try {
                    $env:GH_TOKEN = $stored.Password
                    $authSource = 'credential_manager'
                    Log "Auth: Loaded GH_TOKEN from Windows Credential Manager (QuantumTraderCIWatcher_GH)"
                } catch {
                    Log "Auth: Found stored credential but failed to set GH_TOKEN environment: $($_.Exception.Message)"
                }
            } else {
                Log "Auth: No stored GH token found in Credential Manager (QuantumTraderCIWatcher_GH)"
            }
        } catch {
            Log "Auth: CredentialManager query failed: $($_.Exception.Message)"
        }
    } else {
        Log "Auth: CredentialManager module not available; attempting API-based read of Windows Credential Manager via CredRead (fallback)"
        try {
            if (-not [Type]::GetType('Cred')) {
                $credPinvoke = @"
using System;
using System.Runtime.InteropServices;
using System.Text;
public class Cred {
    [StructLayout(LayoutKind.Sequential, CharSet=CharSet.Unicode)]
    public struct CREDENTIAL {
        public UInt32 Flags;
        public UInt32 Type;
        public string TargetName;
        public string Comment;
        public System.Runtime.InteropServices.ComTypes.FILETIME LastWritten;
        public UInt32 CredentialBlobSize;
        public IntPtr CredentialBlob;
        public UInt32 Persist;
        public UInt32 AttributeCount;
        public IntPtr Attributes;
        public string TargetAlias;
        public string UserName;
    }
    [DllImport("advapi32.dll", SetLastError=true, CharSet=CharSet.Unicode)]
    public static extern bool CredRead(string target, UInt32 type, UInt32 reservedFlag, out IntPtr CredentialPtr);
    [DllImport("advapi32.dll", SetLastError=true)]
    public static extern bool CredFree(IntPtr buffer);
}
"@
                Add-Type -TypeDefinition $credPinvoke -ErrorAction SilentlyContinue
            }
            function Get-StoredCredentialViaCredRead([string]$target) {
                $CRED_TYPE_GENERIC = 1
                $ptr = [IntPtr]::Zero
                $ok = [Cred]::CredRead($target, $CRED_TYPE_GENERIC, 0, [ref]$ptr)
                if (-not $ok) { return $null }
                try {
                    $cred = [System.Runtime.InteropServices.Marshal]::PtrToStructure($ptr, [Type]::GetType('Cred+CREDENTIAL'))
                    $size = $cred.CredentialBlobSize
                    $blob = $cred.CredentialBlob
                    $password = ''
                    if ($size -gt 0 -and $blob -ne [IntPtr]::Zero) {
                        $bytes = New-Object byte[] ($size)
                        [System.Runtime.InteropServices.Marshal]::Copy($blob, $bytes, 0, $size)
                        $password = [System.Text.Encoding]::Unicode.GetString($bytes).TrimEnd([char]0)
                    }
                    $username = $cred.UserName
                    return [PSCustomObject]@{ UserName = $username; Password = $password }
                } finally {
                    [Cred]::CredFree($ptr) | Out-Null
                }
            }

            $cmdkeyTarget = 'QuantumTraderCIWatcher_GH'
            $read = Get-StoredCredentialViaCredRead $cmdkeyTarget
            if ($read -and $read.Password) {
                $env:GH_TOKEN = $read.Password
                $authSource = 'credread'
                Log "Auth: Loaded GH_TOKEN via CredRead fallback (target: $cmdkeyTarget)"
            } else {
                Log "Auth: No credential found via CredRead for target $cmdkeyTarget"
                # Try a cmdkey /list parse to detect whether a credential exists but cannot be read
                try {
                    $cmdOutput = cmd.exe /c 'cmdkey /list' 2>$null
                    if ($cmdOutput) {
                        $joined = $cmdOutput -join "`n"
                        if ($joined -match [regex]::Escape($cmdkeyTarget)) {
                            $authSource = 'cmdkey-present'
                            Log "Auth: cmdkey reports a saved credential with target '$cmdkeyTarget', but the environment cannot read its secret (Add-Type or CredRead blocked)."
                            Log "Auth: To allow automatic loading, install the CredentialManager module or run the installer to persist the token using the module (or run the watcher as a user who can access the secret)."
                        } else {
                            Log "Auth: cmdkey /list did not report a credential with target '$cmdkeyTarget'"
                        }
                    }
                } catch {
                    Log "Auth: cmdkey /list parse fallback failed: $($_.Exception.Message)"
                }
            }
        } catch {
            Log "Auth: CredRead fallback failed: $($_.Exception.Message)"
        }
    }
}

if (-not $env:GH_TOKEN) {
    Log "Auth: No GH token available in environment or Credential Manager. GitHub API calls will fail (HTTP 401)."
} else {
    if (-not $authSource) { $authSource = 'environment' }
    Log "Auth: using GH token from $authSource"
}

Function Get-FailedRuns() {
    # Fetch recent runs as JSON and parse
    $json = gh run list --repo $Repo --limit $MaxRuns --json databaseId,headSha,headBranch,workflowName,status,conclusion,createdAt
    if (-not $json) { return @() }
    $runs = $json | ConvertFrom-Json
    # We consider runs that have conclusion 'failure' or 'cancelled' and workflows related to CI/frontend
    $candidates = $runs | Where-Object { ($_.conclusion -eq 'failure' -or $_.conclusion -eq 'cancelled') -and ($_.workflowName -match 'CI|Frontend') }
    return $candidates
}

Function Get-AuditArtifact([object]$run) {
    $runId = $run.databaseId
    $dir = Join-Path $OutDir $runId
    if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir | Out-Null }

    Log "Inspecting run $runId (workflow: $($run.workflowName), conclusion: $($run.conclusion))"

    # List artifacts for the run using the REST API because `gh run view --json artifacts` is not available
    try {
        $apiResp = gh api -X GET "/repos/$Repo/actions/runs/$runId/artifacts" 2>$null
    } catch {
        $err = $_.Exception.Message
        Log "Failed to call GitHub API for run $($runId): $err"
        return $false
    }
    if (-not $apiResp) {
        Log "No artifacts json returned for run $runId"
        return $false
    }
    $apiObj = $apiResp | ConvertFrom-Json
    $artifacts = $apiObj.artifacts
    if (-not $artifacts -or $artifacts.Count -eq 0) {
        Log "No artifacts found for run $runId"
        return $false
    }

    # Prefer the 'frontend-audit-prod' artifact, otherwise anything with 'audit' in its name
    $target = $artifacts | Where-Object { $_.name -eq 'frontend-audit-prod' } | Select-Object -First 1
    if (-not $target) {
        $target = $artifacts | Where-Object { $_.name -match 'audit' } | Select-Object -First 1
    }
    if (-not $target) {
        Log "No audit-named artifact found for run $runId; skipping"
        return $false
    }

    # Download the artifact into the run directory
    Log "Downloading artifact '$($target.name)' for run $runId into $dir"
    try {
        # Download into a temporary directory then move into place to avoid extraction collisions
        $tempDir = Join-Path $dir ("tmp_dl_" + [System.Guid]::NewGuid().ToString())
        New-Item -ItemType Directory -Path $tempDir | Out-Null
        gh run download $runId --repo $Repo --name $target.name --dir $tempDir | Out-Null
        # Move downloaded files into target dir, overwrite existing files
        Get-ChildItem -Path $tempDir -Recurse -File | ForEach-Object {
            $relative = $_.FullName.Substring($tempDir.Length).TrimStart('\')
            $destPath = Join-Path $dir $relative
            $destDir = Split-Path $destPath -Parent
            if (-not (Test-Path $destDir)) { New-Item -ItemType Directory -Path $destDir | Out-Null }
            Move-Item -Path $_.FullName -Destination $destPath -Force
        }
        Remove-Item -Path $tempDir -Recurse -Force
        Log "Downloaded artifact $($target.name) for run $runId"
        # Save simple metadata
        $meta = [PSCustomObject]@{
            runId = $runId
            headSha = $run.headSha
            headBranch = $run.headBranch
            workflowName = $run.workflowName
            conclusion = $run.conclusion
            artifact = $target.name
            fetchedAt = (Get-Date).ToString("s")
        }
        $meta | ConvertTo-Json | Set-Content -Path (Join-Path $dir "metadata.json")
        # After downloading, look for npm audit JSON files and optionally
        # trigger the audit-fix PR automation if high/critical vulnerabilities are present.
        try {
            $auditFiles = Get-ChildItem -Path $dir -Recurse -Filter 'npm-audit*.json' -File -ErrorAction SilentlyContinue
            if ($auditFiles -and $auditFiles.Count -gt 0) {
                foreach ($f in $auditFiles) {
                    try { $j = Get-Content $f.FullName -Raw | ConvertFrom-Json } catch { $j = $null }
                    if ($j -and $j.metadata -and $j.metadata.vulnerabilities) {
                        $v = $j.metadata.vulnerabilities
                        $high = ($v.high -as [int])
                        $critical = ($v.critical -as [int])
                        if ($high -gt 0 -or $critical -gt 0) {
                            Log "Detected vulnerabilities in run $($runId): high=$high, critical=$critical (file: $($f.FullName))"
                            # Create a GitHub issue so humans can triage the report. This is
                            # the default behavior; automatic PR creation remains opt-in.
                            try {
                                $issueTitle = "[CI] frontend audit: high=$high critical=$critical (run $runId)"
                                $body = @()
                                $body += "Run ID: $runId"
                                $body += "Workflow: $($run.workflowName)"
                                $body += "Head SHA: $($run.headSha)"
                                $body += "Head branch: $($run.headBranch)"
                                $body += "Conclusion: $($run.conclusion)"
                                $body += "Detected at: $((Get-Date).ToString('s'))"
                                $body += ""
                                $body += "Attached artifact: $($f.FullName)"
                                $body += ""
                                $body += "Summary: high=$high, critical=$critical"
                                $bodyPath = Join-Path $dir "issue-body-$runId.md"
                                $body -join "`n" | Set-Content -Path $bodyPath

                                # Use gh to create an issue in the repo. This requires gh to
                                # be authenticated in the environment running the watcher.
                                Log "Creating GitHub issue for run $runId"
                                gh issue create --repo $Repo --title $issueTitle --body-file $bodyPath 2>$null | Out-Null
                                Log "Created GitHub issue for run $runId"
                                # Optionally upload the artifact file as a comment or attach it
                                # to the issue via the web UI; for now we include the artifact path
                                # in the issue body so triagers can download it from the runner.
                                Remove-Item -Path $bodyPath -ErrorAction SilentlyContinue
                            } catch {
                                Log "Failed to create GitHub issue for run $($runId): $($_.Exception.Message)"
                            }
                            # Optionally post a short message to Slack via incoming webhook.
                            try {
                                $webhook = $env:SLACK_WEBHOOK_URL
                                if (-not $webhook -and (Test-Path "$PSScriptRoot\slack_webhook.txt")) {
                                    $webhook = Get-Content (Join-Path $PSScriptRoot 'slack_webhook.txt') -ErrorAction SilentlyContinue | Select-Object -First 1
                                }
                                if ($webhook) {
                                    $text = "[CI] frontend audit alert: run $runId — high=$high critical=$critical — headSha=$($run.headSha)"
                                    $payload = @{ text = $text }
                                    Invoke-RestMethod -Uri $webhook -Method Post -Body ($payload | ConvertTo-Json -Depth 3) -ContentType 'application/json' -ErrorAction Stop
                                    Log "Posted Slack notification for run $runId"
                                }
                            } catch {
                                Log "Failed to post Slack notification for run $($runId): $($_.Exception.Message)"
                            }

                            # Controlled auto-PR behavior: requires env var AUTO_CREATE_AUDIT_PR=1 or presence of scripts/auto_create_pr file
                            $autoEnabled = $false
                            if ($env:AUTO_CREATE_AUDIT_PR -eq '1') { $autoEnabled = $true }
                            if (Test-Path "$PSScriptRoot\auto_create_pr") { $autoEnabled = $true }
                            if ($autoEnabled) {
                                Log "AUTO_CREATE_AUDIT_PR enabled; invoking create-audit-fix-pr for run $runId"
                                try {
                                    # If operator wants auto-merge, create a small marker file that the PR script checks
                                    if ($env:AUTO_ADD_AUTO_MERGE -eq '1') { New-Item -Path (Join-Path $PSScriptRoot 'auto_merge') -ItemType File -Force | Out-Null }
                                    pwsh -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot 'create-audit-fix-pr.ps1') -RunId $runId -AutoOpen
                                    if (Test-Path (Join-Path $PSScriptRoot 'auto_merge')) { Remove-Item (Join-Path $PSScriptRoot 'auto_merge') -Force -ErrorAction SilentlyContinue }
                                } catch {
                                    Log "Failed to invoke PR creation script for run $($runId): $($_.Exception.Message)"
                                }
                            } else {
                                Log "AUTO_CREATE_AUDIT_PR not enabled; skipping automatic PR creation for run $runId"
                            }
                        }
                    }
                }
            }
        } catch {
            Log "Error while checking audit files for run $($runId): $($_.Exception.Message)"
        }
        return $true
    } catch {
        $err = $_.Exception.Message
    Log "Failed to download artifact for run $($runId): $err"
        return $false
    }
}

# Main loop
while ($true) {
    $failed = Get-FailedRuns
    if ($failed.Count -eq 0) {
        Log "No recent failed CI/frontend runs found"
    } else {
        foreach ($r in $failed) {
            try {
                $ok = Get-AuditArtifact $r
                if ($ok) { Log "Artifact saved for run $($r.databaseId)" }
            } catch {
                $err = $_.Exception.Message
                Log "Error handling run $($r.databaseId): $err"
            }
        }
    }
    if ($Once) { break }
    Start-Sleep -Seconds $PollIntervalSeconds
}

Log "Watcher finished"