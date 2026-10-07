#ifndef AppVersion
  #define AppVersion "0.4.0"
#endif
#ifndef AppDistDir
  #define AppDistDir "..\work\dist\app"
#endif
#ifndef ReleaseDir
  #define ReleaseDir "..\output"
#endif

#define AppLongName "AI换脸涉案图像智能鉴伪与原始素材溯源系统"
#define AppShortName "换脸寻源系统 v0.4.0"
#define AppExeName "换脸寻源系统_v0.4.0.exe"
#define AppPublisher "换脸寻源项目组"

[Setup]
AppId={{BA84DD0A-D32A-4A30-88B1-1546A06EE248}
AppName={#AppLongName}
AppVerName={#AppLongName} {#AppVersion}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
AppCopyright=Copyright (C) 2026 {#AppPublisher}
AppComments={#AppShortName}
DefaultDirName={localappdata}\Programs\{#AppShortName}
DefaultGroupName={#AppShortName}
DisableProgramGroupPage=yes
AllowNoIcons=no
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
OutputDir={#ReleaseDir}
OutputBaseFilename=换脸寻源系统_v0.4.0_Setup
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
SetupLogging=yes
UsePreviousAppDir=yes
UsePreviousGroup=yes
UsePreviousTasks=yes
CloseApplications=yes
CloseApplicationsFilter={#AppExeName},identity_worker.exe,source_search.exe,python.exe
RestartApplications=no
Uninstallable=yes
UninstallDisplayName={#AppLongName} {#AppVersion}
UninstallDisplayIcon={app}\{#AppExeName}
VersionInfoVersion=0.4.0.0
VersionInfoProductVersion=0.4.0.0
VersionInfoProductName={#AppLongName}
VersionInfoDescription={#AppShortName} 安装程序
VersionInfoCompany={#AppPublisher}
VersionInfoCopyright=Copyright (C) 2026 {#AppPublisher}

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "创建桌面快捷方式"; GroupDescription: "快捷方式："; Flags: checkedonce

[Files]
Source: "{#AppDistDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\{#AppShortName}\{#AppShortName}"; Filename: "{app}\{#AppExeName}"; WorkingDir: "{app}"; Comment: "{#AppLongName}"
Name: "{autoprograms}\{#AppShortName}\停止{#AppShortName}"; Filename: "{sys}\WindowsPowerShell\v1.0\powershell.exe"; Parameters: "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File ""{app}\stop_system.ps1"""; WorkingDir: "{app}"
Name: "{autodesktop}\{#AppShortName}"; Filename: "{app}\{#AppExeName}"; WorkingDir: "{app}"; Comment: "{#AppLongName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExeName}"; Description: "启动 {#AppShortName}"; WorkingDir: "{app}"; Flags: nowait postinstall skipifsilent

[UninstallRun]
Filename: "{sys}\WindowsPowerShell\v1.0\powershell.exe"; Parameters: "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File ""{app}\stop_system.ps1"" -Silent"; WorkingDir: "{app}"; Flags: runhidden waituntilterminated skipifdoesntexist; RunOnceId: "StopFaceSourceTraceV040"
