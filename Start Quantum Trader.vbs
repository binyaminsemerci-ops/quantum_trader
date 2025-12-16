' Quantum Trader - VBScript Launcher
' ====================================
' Double-click this file to start Quantum Trader
' Creates a desktop shortcut on first run

Set WshShell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

' Get script directory
ScriptDir = fso.GetParentFolderName(WScript.ScriptFullName)

' Create desktop shortcut if it doesn't exist
DesktopPath = WshShell.SpecialFolders("Desktop")
ShortcutPath = DesktopPath & "\Quantum Trader.lnk"

If Not fso.FileExists(ShortcutPath) Then
    Set Shortcut = WshShell.CreateShortcut(ShortcutPath)
    Shortcut.TargetPath = "powershell.exe"
    Shortcut.Arguments = "-ExecutionPolicy Bypass -File """ & ScriptDir & "\start_quantum_trader.ps1"""
    Shortcut.WorkingDirectory = ScriptDir
    Shortcut.IconLocation = "powershell.exe,0"
    Shortcut.Description = "Start Quantum Trader AI Trading System"
    Shortcut.Save
    
    MsgBox "Desktop shortcut created!" & vbCrLf & vbCrLf & _
           "Starting Quantum Trader...", vbInformation, "Quantum Trader"
End If

' Start the PowerShell script
WshShell.Run "powershell.exe -ExecutionPolicy Bypass -File """ & ScriptDir & "\start_quantum_trader.ps1""", 1, False

Set WshShell = Nothing
Set fso = Nothing
