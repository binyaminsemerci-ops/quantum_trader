# Summarize downloaded npm audit artifacts under artifacts/ci-runs
$repo='binyaminsemerci-ops/quantum_trader'
$base='artifacts/ci-runs'
$summary=@()
Get-ChildItem -Directory $base | ForEach-Object {
    $dir=$_.FullName
    $metaFile=Join-Path $dir 'metadata.json'
    if(-not (Test-Path $metaFile)){ return }
    $meta=Get-Content $metaFile | ConvertFrom-Json
    $runId=$meta.runId
    $artifactName=$meta.artifact
    try{ gh run download $runId --repo $repo --name $artifactName --dir $dir 2>$null } catch { }
    $files=Get-ChildItem -Path $dir -Recurse -Filter 'npm-audit*.json' -File -ErrorAction SilentlyContinue
    if(-not $files -or $files.Count -eq 0){
        $summary += [pscustomobject]@{runId=$runId; artifact=$artifactName; found=$false; high=0; critical=0; total=0; file=''}
        return
    }
    foreach($f in $files){
        try{ $j=Get-Content $f.FullName -Raw | ConvertFrom-Json } catch { $j=$null }
        if(-not $j){
            $summary += [pscustomobject]@{runId=$runId; artifact=$artifactName; found=$false; high=0; critical=0; total=0; file=$f.FullName}
            continue
        }
        $v=$j.metadata.vulnerabilities
        $summary += [pscustomobject]@{runId=$runId; artifact=$artifactName; found=$true; high = ($v.high -as [int]); critical = ($v.critical -as [int]); total = ($v.total -as [int]); file = $f.FullName }
    }
}
$outFile = Join-Path $base 'summary.json'
$summary | ConvertTo-Json -Depth 4 | Set-Content -Path $outFile
$summary | Format-Table -AutoSize
Write-Host "Wrote $outFile"
