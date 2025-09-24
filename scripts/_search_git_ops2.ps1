$patterns = @('git push','git commit','git push --set-upstream','git push origin')
Get-ChildItem -Path 'api_bodies' -Recurse -File | ForEach-Object {
    $file = $_.FullName
    try {
        $content = Get-Content -LiteralPath $file -ErrorAction Stop
    } catch { continue }
    for ($i=0; $i -lt $content.Count; $i++) {
        foreach ($p in $patterns) {
            if ($content[$i] -match [regex]::Escape($p)) {
                Write-Output ("MATCH: {0}:{1}: {2}" -f $file, ($i+1), $content[$i].Trim())
                $start = [Math]::Max(0, $i-3)
                $end = [Math]::Min($content.Count-1, $i+3)
                for ($j=$start; $j -le $end; $j++) { Write-Output ("{0,6}: {1}" -f ($j+1), $content[$j]) }
                Write-Output '---'
            }
        }
    }
}