param(
    [string]$CommitMessage = "Oppdateringer",
    [string]$Remote = "origin",
    [string]$Branch = "main"
)

Write-Host "Stager alle endringer …"
git add -A

Write-Host "Committer …"
git commit -m $CommitMessage

Write-Host "Pusher …"
git push $Remote $Branch
