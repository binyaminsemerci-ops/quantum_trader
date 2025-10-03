param(
  [int]$Port = 40111
)
$ErrorActionPreference='Stop'
try {
  $r = Invoke-WebRequest -UseBasicParsing ("http://127.0.0.1:" + $Port + "/")
  "FETCH_OK Status=$($r.StatusCode) Len=$($r.Content.Length)"
  if ($r.Content) {"FETCH_BODY_START"; $r.Content; "FETCH_BODY_END"}
} catch {
  "FETCH_ERR " + $_.Exception.GetType().Name + ": " + $_.Exception.Message
  exit 2
}
