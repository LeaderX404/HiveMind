; HiveMind installer script (issue #42, wave #42).
; Stage 3 starter: compile with Inno Setup after PyInstaller
; produces dist/HiveMind.exe -  iscc installer\hivemind.iss
; Not wired into CI yet: the next stage compiles Setup.exe in
; the build workflow (choco install innosetup) and attaches it
; to the release next to the bare exe.

[Setup]
AppId={{1A2B3C4D-5E6F-4A7B-8C9D-0E1F2A3B4C5D}
AppName=HiveMind
AppVersion=0.1.0
AppPublisher=LeaderX404
DefaultDirName={localappdata}\HiveMind
DefaultGroupName=HiveMind
OutputBaseFilename=HiveMind-Setup
PrivilegesRequired=lowest
Compression=lzma2
SolidCompression=yes
ArchitecturesInstallIn64BitMode=x64compatible
DisableProgramGroupPage=yes

[Files]
Source: "..\dist\HiveMind.exe"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{userdesktop}\HiveMind"; Filename: "{app}\HiveMind.exe"
Name: "{group}\HiveMind"; Filename: "{app}\HiveMind.exe"

[UninstallDelete]
Type: filesandordirs; Name: "{app}"
