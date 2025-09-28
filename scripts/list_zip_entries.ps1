param(
    [string]$ZipPath
)
Add-Type -AssemblyName System.IO.Compression.FileSystem
$zip = [System.IO.Compression.ZipFile]::OpenRead($ZipPath)
foreach($e in $zip.Entries){
    Write-Host $e.FullName
}
$zip.Dispose()
