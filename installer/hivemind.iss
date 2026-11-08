; HiveMind installer (issues #43, #44, wave #47).
;
; Installs the ONEDIR app: the app tree - both executables
; plus _internal - goes to {app}\app, and the heavy core
; bundle is downloaded ONCE, verified with SHA-256 before
; extraction, into {app}\core. No admin rights anywhere.
; CI compiles this script with /DMyAppVersion, /DCoreUrl and
; /DCoreSha256 so every release verifies exactly the archive
; it published.
#ifndef MyAppVersion
#define MyAppVersion "0.1.4"
#endif
#ifndef CoreUrl
#define CoreUrl ""
#endif
#ifndef CoreSha256
#define CoreSha256 ""
#endif

[Setup]
AppId={{1A2B3C4D-5E6F-4A7B-8C9D-0E1F2A3B4C5D}
AppName=HiveMind
AppVersion={#MyAppVersion}
AppPublisher=HiveMind
DefaultDirName={localappdata}\Programs\HiveMind
DefaultGroupName=HiveMind
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=Output
OutputBaseFilename=HiveMind-Setup

[Files]
; The onedir app tree: both executables plus _internal
; (wave #47). Source: dist/HiveMind-UI built by CI.
Source: "..\dist\HiveMind-UI\*"; DestDir: "{app}\app";     Flags: recursesubdirs ignoreversion
Source: "fetch_core.ps1"; DestDir: "{tmp}"; Flags: deleteafterinstall

[Dirs]
Name: "{app}\core"

[Icons]
Name: "{group}\HiveMind"; Filename: "{app}\app\HiveMind.exe"
Name: "{group}\HiveMind Debug Console"; Filename: "{app}\app\HiveMind-Debug.exe"
Name: "{commondesktop}\HiveMind"; Filename: "{app}\app\HiveMind.exe";     Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut";     Flags: checkedonce

[Run]
Filename: "{app}\app\HiveMind.exe"; Description: "Launch HiveMind";     Flags: nowait postinstall skipifsilent

[UninstallDelete]
; The core bundle lives outside the app tree: remove the
; whole install directory - {app}\app and {app}\core - on
; uninstall, a clean slate with no orphaned gigabytes.
Type: filesandordirs; Name: "{app}"

[Code]
const
  CoreUrl = '{#CoreUrl}';
  CoreSha256 = '{#CoreSha256}';

procedure CurStepChanged(CurStep: TSetupStep);
var
  ResultCode: Integer;
begin
  if CurStep <> ssPostInstall then exit;
  if (CoreUrl = '') or (CoreSha256 = '') then begin
    MsgBox('The core bundle URL was not defined at compile ' +
           'time - the app will download the core on first ' +
           'run instead.', mbInformation, MB_OK);
    exit;
  end;
  if DirExists(ExpandConstant('{app}\core\torch')) then begin
    MsgBox('The core is already installed - reusing it, the ' +
           'core downloads once and not per release.',
           mbInformation, MB_OK);
    exit;
  end;
  if MsgBox('The heavy core - PyTorch, transformers - is not ' +
            'installed yet. Download it now? About 1 GB, ' +
            'verified with SHA-256 before extraction.',
            mbConfirmation, MB_YESNO) = IDNO then
  begin
    MsgBox('The app will offer to download the core on first ' +
           'run instead.', mbInformation, MB_OK);
    exit;
  end;
  if Exec(ExpandConstant('{sys}\WindowsPowerShell\v1.0\powershell.exe'),
      '-ExecutionPolicy Bypass -File "' + ExpandConstant('{tmp}\fetch_core.ps1') +
      '" -Url "' + CoreUrl + '" -Sha256 ' + CoreSha256 +
      ' -Dest "' + ExpandConstant('{app}\core') + '"',
      '', SW_SHOW, ewWaitUntilTerminated, ResultCode) then
  begin
    if ResultCode = 0 then
      MsgBox('The core is installed and verified.', mbInformation, MB_OK)
    else
      MsgBox('The core download failed with exit code ' +
             IntToStr(ResultCode) + '. The app will retry on ' +
             'first run.', mbError, MB_OK);
  end else
    MsgBox('PowerShell could not start - the app will download ' +
           'the core on first run.', mbError, MB_OK);
end;
