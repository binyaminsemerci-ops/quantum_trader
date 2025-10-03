<#
install_nssm_and_configure_service.ps1

Runs as Administrator. Creates scripts\tools, downloads NSSM (if missing), installs/updates
QuantumTraderCIWatcher service (NSSM), optionally configures service account and sets
GH_TOKEN into the service environment using NSSM's AppEnvironmentExtra.

Usage (run as Administrator):
  pwsh -NoProfile -ExecutionPolicy Bypass -File .\scripts\install_nssm_and_configure_service.ps1 \
    [-NssmUrl <url>] [-ServiceName <name>] [-ServiceUser <DOMAIN\\user>] [-ServicePassword <pass>] [-GHToken <token>] [-StartService]

Notes:
 - Supplying ServicePassword on the command-line may leave traces in cmd history; prefer using an interactive prompt or using a vault.
 - If you prefer the service to run as a specific user, provide ServiceUser and ServicePassword.
 - If you provide GHToken, it will be injected into the service via NSSM AppEnvironmentExtra. The token will not be echoed to the console.
 - This script will not echo GHToken to output.
#>
param(
    [string]$NssmUrl = 'https://nssm.cc/release/nssm-2.24.zip',
    [string]$ServiceName = 'QuantumTraderCIWatcher',
    [string]$ServiceUser = '',
    [System.Security.SecureString]$ServicePassword = $null,
    [string]$GHToken = '',
    [bool]$StartService = $true,
    [bool]$PersistTokenToCredentialManager = $true,
    [string]$CredentialTarget = '',
    [switch]$AlsoSetEnv
)

function Assert-Admin {
    $isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
    if (-not $isAdmin) {
        Write-Error "This script must be run as Administrator. Please re-run in an elevated PowerShell."; exit 1
    }
}

Assert-Admin

$toolsDir = Join-Path $PSScriptRoot 'tools'
if (-not (Test-Path $toolsDir)) { New-Item -ItemType Directory -Path $toolsDir | Out-Null }

# Try to find existing nssm
$nssmLocal = Join-Path $toolsDir 'nssm.exe'
$nssm = $null
if (Test-Path $nssmLocal) { $nssm = (Resolve-Path $nssmLocal).ProviderPath }
else {
    try { $p = (Get-Command nssm -ErrorAction SilentlyContinue).Source; if ($p) { $nssm = $p } } catch {}
}

if (-not $nssm) {
    Write-Host "nssm.exe not found locally; downloading from $NssmUrl into $toolsDir"
    $zipPath = Join-Path $toolsDir 'nssm.zip'
    Invoke-WebRequest -Uri $NssmUrl -OutFile $zipPath -UseBasicParsing
    Write-Host "Downloaded NSSM to $zipPath"
    Expand-Archive -Path $zipPath -DestinationPath $toolsDir -Force
    # find win64 nssm.exe inside extracted folder
    $extracted = Get-ChildItem -Path $toolsDir -Directory | Where-Object { $_.Name -match 'nssm' } | Select-Object -First 1
    if ($extracted) {
        $candidate = Join-Path $extracted.FullName 'win64\nssm.exe'
        if (-not (Test-Path $candidate)) {
            # older zips may have different layout; try search
            $candidate = Get-ChildItem -Path $toolsDir -Filter 'nssm.exe' -Recurse -File | Where-Object { $_.FullName -match 'win64' } | Select-Object -First 1
        }
        if ($candidate) {
            Copy-Item -Force -Path $candidate.FullName -Destination $nssmLocal
            Write-Host "Copied nssm.exe to $nssmLocal"
        } else {
            # fallback: copy any nssm.exe
            $any = Get-ChildItem -Path $toolsDir -Recurse -Filter 'nssm.exe' -File | Select-Object -First 1
            if ($any) { Copy-Item -Force -Path $any.FullName -Destination $nssmLocal; Write-Host "Copied nssm.exe to $nssmLocal (fallback)" }
        }
    }
    Remove-Item -Path $zipPath -Force -ErrorAction SilentlyContinue
    $nssm = $nssmLocal
}

if (-not (Test-Path $nssm)) { Write-Error "nssm.exe not found or failed to install into $toolsDir"; exit 2 }

# Prepare pwsh executable and watcher path
$pwshPath = (Get-Command pwsh -ErrorAction SilentlyContinue).Source
if (-not $pwshPath) { $pwshPath = 'pwsh' }

# Resolve the watcher script path relative to this installer script directory ($PSScriptRoot)
$candidate = Join-Path $PSScriptRoot 'ci-watch.ps1'
if (Test-Path $candidate) {
    $watcherPath = (Resolve-Path $candidate).ProviderPath
} else {
    # Fallback: try repo root (parent of scripts) in case the script was invoked from elsewhere
    $repoRootCandidate = Join-Path (Split-Path -Parent $PSScriptRoot) 'ci-watch.ps1'
    if (Test-Path $repoRootCandidate) {
        $watcherPath = (Resolve-Path $repoRootCandidate).ProviderPath
    } else {
        Write-Error "ci-watch.ps1 not found relative to $PSScriptRoot or its parent. Please ensure scripts/ci-watch.ps1 exists."; exit 4
    }
}

$pwshArgs = '-NoProfile -ExecutionPolicy Bypass -File "' + $watcherPath + '" -PollIntervalSeconds 300'

Write-Host "Using nssm: $nssm"
Write-Host "Installing/updating service $ServiceName -> $pwshPath $pwshArgs"

# Install or update the service via NSSM
try {
    & $nssm install $ServiceName $pwshPath $pwshArgs 2>$null | Out-Null
} catch {
    # ignore install error if already exists; we'll continue to set params
}

# Set service to auto start
& $nssm set $ServiceName Start SERVICE_AUTO_START

# If ServiceUser provided, configure service to run as that user. Use sc to set credentials (safer than embedding in a command string).
if ($ServiceUser -ne '') {
    # Ensure we have a SecureString for the password; prompt if missing
    if (-not $ServicePassword) {
        Write-Host "Service user specified but no password provided. Prompting for password (will not echo)."
        $sec = Read-Host -AsSecureString "Password for $ServiceUser"
        $ServicePassword = $sec
    }
    # Convert SecureString to plain for sc.exe invocation (sc.exe needs plaintext password)
    try {
        $plainPtr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($ServicePassword)
        $plainPwd = [Runtime.InteropServices.Marshal]::PtrToStringAuto($plainPtr)
        [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($plainPtr) | Out-Null
    } catch {
        Write-Error "Failed to convert SecureString to plain text: $($_.Exception.Message)"; exit 3
    }
    Write-Host "Configuring service to run as $ServiceUser"
    # sc config requires a space after the argument name
    sc.exe config $ServiceName obj= "$ServiceUser" password= "$plainPwd" | Out-Null
    # Clear plaintext password variable as soon as possible
    Remove-Variable plainPwd -ErrorAction SilentlyContinue
}

# If GHToken provided, inject into service environment via NSSM AppEnvironmentExtra
# If GHToken not provided on the command-line, offer a secure prompt (interactive)
if ($GHToken -eq '') {
    try {
        Write-Host "No GHToken provided on the command line. You may enter one now (will not be echoed), or press Enter to skip."
        $ghSecure = Read-Host -AsSecureString "GH token (press Enter to skip)"
        if ($ghSecure -and $ghSecure.Length -gt 0) {
            $ptr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($ghSecure)
            $GHToken = [Runtime.InteropServices.Marshal]::PtrToStringAuto($ptr)
            [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($ptr) | Out-Null
        }
    } catch {
        # non-interactive hosts may throw; ignore and continue
    }
}

if ($GHToken -ne '') {
    # Prefer to persist token to Windows Credential Manager (LocalMachine) for services
    $storedOk = $false
    if ($PersistTokenToCredentialManager) {
        try {
            $storedOk = $false
            if (-not (Get-Module -ListAvailable -Name CredentialManager)) {
                Write-Host "CredentialManager module not found; attempting to install it for CurrentUser"
                try {
                    Install-Module -Name CredentialManager -Scope CurrentUser -Force -ErrorAction Stop
                } catch {
                    Write-Warning "Failed to install CredentialManager module: $($_.Exception.Message)"
                }
            }
            Import-Module CredentialManager -ErrorAction Stop
            # Determine target name
            if ([string]::IsNullOrEmpty($CredentialTarget)) { $CredentialTarget = "$ServiceName`_GH" }
            # Use New-StoredCredential to persist under LocalMachine so LocalSystem can access it
            New-StoredCredential -Target $CredentialTarget -Username 'gh' -Password $GHToken -Persist LocalMachine -ErrorAction Stop
            Write-Host "Stored GH token in Windows Credential Manager (Target: $CredentialTarget)"
            $storedOk = $true
        } catch {
            Write-Warning "Failed to persist token into Credential Manager via module: $($_.Exception.Message)"
            $storedOk = $false
            # Fallback: try using cmdkey.exe to add a generic credential so we don't require the PSGallery module
            try {
                Write-Host "Attempting fallback persistence using cmdkey.exe (will store under target: $ServiceName`_GH)"
                if ([string]::IsNullOrEmpty($CredentialTarget)) { $CredentialTarget = "$ServiceName`_GH" }
                $targetName = $CredentialTarget
                # cmdkey: /add:<TargetName> /user:<User> /pass:<Password>
                $escapedPass = $GHToken -replace '"', '""'
                $cmd = "cmdkey /add:`"$targetName`" /user:`"gh`" /pass:`"$escapedPass`""
                cmd.exe /c $cmd
                Write-Host "Stored GH token using cmdkey under target: $targetName"
                $storedOk = $true
            } catch {
                Write-Warning "Fallback persistence using cmdkey failed: $($_.Exception.Message)"
                $storedOk = $false
            }
        }
    }

    if ($AlsoSetEnv -or -not $PersistTokenToCredentialManager -or -not $storedOk) {
        Write-Host "Adding GH_TOKEN to service environment (not echoed)"
        & $nssm set $ServiceName AppEnvironmentExtra "GH_TOKEN=$GHToken"
    } else {
        Write-Host "GH token persisted to Credential Manager; clearing NSSM AppEnvironmentExtra so service uses the vaulted token."
        try {
            & $nssm reset $ServiceName AppEnvironmentExtra
            Write-Host "Cleared NSSM AppEnvironmentExtra for service $ServiceName"
        } catch {
            Write-Warning "Failed to clear NSSM AppEnvironmentExtra: $($_.Exception.Message)"
        }
    }
}

# Try starting the service
try {
    & $nssm start $ServiceName
    Start-Sleep -Seconds 2
    # query status
    $status = & $nssm status $ServiceName
    Write-Host "NSSM status: $status"
} catch {
    Write-Warning "Failed to start service via NSSM: $($_.Exception.Message)"
}

Write-Host "Done. Verify the service is running and check artifacts/ci-runs/monitor.log for activity."
Write-Host "If GH_TOKEN was injected, GH will use it for gh CLI operations."

# Helpful post-install verification instructions
Write-Host "\nPost-install checks (run manually):"
Write-Host "  Get-Service -Name $ServiceName | Format-List *"
Write-Host "  C:\\quantum_trader\\scripts\\tools\\nssm.exe status $ServiceName"
Write-Host "  Get-Content .\\artifacts\\ci-runs\\monitor.log -Tail 80"

# End
