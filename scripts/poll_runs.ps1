$ids = @(18066528875,18066528880,18066528876,18066528884)
function Wait-Run($id){
    for ($i=0; $i -lt 60; $i++){
        $json = gh run view $id --repo binyaminsemerci-ops/quantum_trader --json databaseId,name,status,conclusion -q '{id: .databaseId, name: .name, status: .status, conclusion: .conclusion}'
        $o = $json | ConvertFrom-Json
        Write-Host ("$($o.id) $($o.name) -> status=$($o.status) conclusion=$($o.conclusion)")
        if($o.status -eq 'completed'){
            return $o
        }
        Start-Sleep -Seconds 6
    }
    return $null
}

foreach($id in $ids){
    Write-Host "--- Waiting for run $id ---"
    $res = Wait-Run $id
    if($res -eq $null){ Write-Host "Timed out waiting for $id" }
}
