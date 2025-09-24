$patterns = @('git push','git commit','git push --set-upstream','git push origin')
Get-ChildItem -Path 'api_bodies' -Recurse -File | ForEach-Object {
    try {
        $matches = Select-String -Path $_.FullName -Pattern ($patterns -join '|') -AllMatches -SimpleMatch
    } catch { $matches = $null }
    if ($matches) {
        foreach ($m in $matches) {
            Write-Output "MATCH: $($_.FullName):$($m.LineNumber): $($m.Line.Trim())"
            $lines = Get-Content -LiteralPath $_.FullName
            $start = [Math]::Max(0, $m.LineNumber - 3)
            $end = [Math]::Min($lines.Count - 1, $m.LineNumber + 1)
            for ($i = $start; $i -le $end; $i++) {
                Write-Output ("{0,6}: {1}" -f ($i + 1), $lines[$i])
            }
            Write-Output '---'
        }
    }
}