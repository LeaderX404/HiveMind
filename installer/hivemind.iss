; HiveMind installer script (issues #43/#44, waves #42-#43).
; Compiled in CI: choco install innosetup, then
;   iscc /DMyAppVersion=<x.y.z> installer\hivemind.iss
; CI passes the version from RELEASE.md so the Setup.exe always
; matches the release it ships with. Output goes to Output/.

#ifndef MyAppVersion
#define MyAppVersion "0.1.1"
#endif

[Setup]
AppId={{1A2B3C4D-5E6F-4A7B-8C9D-0E1F2A3B4C5D}
AppName=HiveMind
AppVersion={#MyAppVersion}
AppPublisher=LeaderX404
DefaultDirName={localappdata}\Programs\HiveMind
DefaultGroupName=HiveMind
OutputDir=Output
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

[Run]
Filename: "{app}\HiveMind.exe"; Description: "Launch HiveMind"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
Type: filesandordirs; Name: "{app}"
