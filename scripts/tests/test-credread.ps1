<#
Unit-style test to exercise CredRead fallback in a local, script-only manner.
This script:
 - Adds a temporary credential via cmdkey
 - Invokes the CredRead helper (Add-Type) to read it
 - Verifies the password matches
 - Cleans up the credential via cmdkey /delete
Note: This test does not call GitHub or network services.
#>

param(
    [string]$Target = 'QuantumTraderCIWatcher_GH',
    # The password here is test-only. If detect-secrets flags it as a false
    # positive you can allowlist it in scan configuration. Keep entropy low for
    # test data to avoid high-entropy detectors.
    [string]$Password = 'dev-test-token' # pragma: allowlist secret
)

Function Log([string]$m) { Write-Host "[test-credread] $m" }

# Define CredRead/CredWrite/CredDelete helpers via Add-Type so we can create and read a credential in-process
 $credHelper = @"
using System;
using System.Runtime.InteropServices;
using System.Text;
public class CredHelper {
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
    [DllImport("advapi32.dll", SetLastError=true, CharSet=CharSet.Unicode)]
    public static extern bool CredWrite(ref CREDENTIAL userCredential, UInt32 flags);
    [DllImport("advapi32.dll", SetLastError=true, CharSet=CharSet.Unicode)]
    public static extern bool CredDelete(string target, UInt32 type, UInt32 flags);

    public static bool Write(string target, string username, string secret, uint persist) {
        byte[] bytes = Encoding.Unicode.GetBytes(secret + "\0");
        IntPtr ptr = Marshal.AllocHGlobal(bytes.Length);
        try {
            Marshal.Copy(bytes, 0, ptr, bytes.Length);
            CREDENTIAL c = new CREDENTIAL();
            c.Flags = 0;
            c.Type = 1; // GENERIC
            c.TargetName = target;
            c.Comment = null;
            c.CredentialBlobSize = (uint)bytes.Length;
            c.CredentialBlob = ptr;
            c.Persist = persist;
            c.AttributeCount = 0;
            c.Attributes = IntPtr.Zero;
            c.TargetAlias = null;
            c.UserName = username;
            return CredWrite(ref c, 0);
        } finally {
            if (ptr != IntPtr.Zero) Marshal.FreeHGlobal(ptr);
        }
    }

    public static string Read(string target) {
        IntPtr p;
        if (!CredRead(target, 1, 0, out p)) return null;
        try {
            CREDENTIAL c = (CREDENTIAL)Marshal.PtrToStructure(p, typeof(CREDENTIAL));
            if (c.CredentialBlobSize == 0 || c.CredentialBlob == IntPtr.Zero) return null;
            byte[] b = new byte[c.CredentialBlobSize];
            Marshal.Copy(c.CredentialBlob, b, 0, (int)c.CredentialBlobSize);
            string s = Encoding.Unicode.GetString(b).TrimEnd('\0');
            return s;
        } finally {
            CredFree(p);
        }
    }

    public static bool Delete(string target) {
        return CredDelete(target, 1, 0);
    }
}
"@
Add-Type -TypeDefinition $credHelper -ErrorAction Stop

Log "Writing temporary credential via CredHelper.Write for target $Target"
$writeOk = [CredHelper]::Write($Target, 'gh', $Password, 1)
if (-not $writeOk) { throw "CredHelper.Write failed" }
Start-Sleep -Milliseconds 200

function Get-StoredCredentialViaCredRead([string]$target) {
    # Use the CredHelper wrapper to read the secret created by CredHelper.Write
    try {
        $pwd = [CredHelper]::Read($target)
        if ($pwd) { return [PSCustomObject]@{ UserName = 'gh'; Password = $pwd } }
        return $null
    } catch {
        return $null
    }
}

Log "Invoking CredRead on target $Target"
$read = Get-StoredCredentialViaCredRead $Target
if ($read -and $read.Password) {
    if ($read.Password -eq $Password) {
        Log "SUCCESS: CredRead returned expected password"
        $rc = 0
    } else {
        Log "FAIL: CredRead returned a password but it does not match (got: $($read.Password))"
        $rc = 2
    }
} else {
    Log "FAIL: CredRead did not return a password or returned null"
    $rc = 3
}

# Cleanup using CredHelper.Delete
Log "Cleaning up credential"
[CredHelper]::Delete($Target) | Out-Null

exit $rc
