function Test-GhAvailable {
    param(
        [switch]$RequireAuth
    )
    $ghExe = Get-Command gh -ErrorAction SilentlyContinue
    if (-not $ghExe) {
        Write-Error "gh CLI not found in PATH. Install GitHub CLI: https://cli.github.com/"
        return $false
    }
    if ($RequireAuth) {
        if (-not $env:GH_TOKEN) {
            # Try to detect if gh is authenticated via `gh auth status`
            try {
                gh auth status 2>$null
                if ($LASTEXITCODE -ne 0) {
                    Write-Error "gh CLI not authenticated. Run 'gh auth login' or set GH_TOKEN in environment."
                    return $false
                }
            } catch {
                Write-Error "Unable to verify gh auth status: $($_.Exception.Message)"
                return $false
            }
        }
    }
    return $true
}

Export-ModuleMember -Function Test-GhAvailable
