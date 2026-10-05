#define AppLongName "换脸寻源——AI换脸涉案图像智能鉴伪与原始素材溯源系统"
#define AppShortName "换脸寻源双搜索系统"
#define AppVersion "1.0.0"
#define AppExeName "换脸寻源双搜索系统.exe"
#define AppPublisher "换脸寻源项目组"

#ifndef AppDistDir
  #define AppDistDir "..\dist\app"
#endif

#ifndef ReleaseDir
  #define ReleaseDir "..\..\release_v1.0"
#endif

[Setup]
AppId={{2A96BB7E-3574-4A10-9CF5-E4B3E67494DA}
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
OutputBaseFilename=换脸寻源_双搜索系统_Setup_v1.0
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
SetupLogging=yes
UsePreviousAppDir=yes
UsePreviousGroup=yes
UsePreviousTasks=yes
CloseApplications=yes
RestartApplications=no
Uninstallable=yes
UninstallDisplayName={#AppLongName}
UninstallDisplayIcon={app}\{#AppExeName}
VersionInfoVersion=1.0.0.0
VersionInfoProductVersion=1.0.0.0
VersionInfoProductName={#AppLongName}
VersionInfoDescription={#AppShortName} 安装程序
VersionInfoCompany={#AppPublisher}
VersionInfoCopyright=Copyright (C) 2026 {#AppPublisher}

[Languages]
; Default.isl is shipped with every official Inno Setup installation.  The
; application/task labels below remain Chinese without relying on an optional
; third-party language pack.
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "创建桌面快捷方式"; GroupDescription: "快捷方式："; Flags: checkedonce

[Files]
Source: "{#AppDistDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\{#AppShortName}\{#AppShortName}"; Filename: "{app}\{#AppExeName}"; WorkingDir: "{app}"; Comment: "{#AppLongName}"
Name: "{autodesktop}\{#AppShortName}"; Filename: "{app}\{#AppExeName}"; WorkingDir: "{app}"; Comment: "{#AppLongName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExeName}"; Description: "启动 {#AppShortName}"; WorkingDir: "{app}"; Flags: nowait postinstall skipifsilent
