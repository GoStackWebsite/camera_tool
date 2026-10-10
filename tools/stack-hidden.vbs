' Runs the Congruence Eye stack supervisor (tools\stack.ps1 start) without any console window.
' Used by the "Congruence Eye Stack" logon task; double-clicking it works too.
Set sh = CreateObject("WScript.Shell")
dir = Left(WScript.ScriptFullName, InStrRev(WScript.ScriptFullName, "\") - 1)
sh.Run "powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File """ & dir & "\stack.ps1"" start", 0, False
