#!/usr/bin/env bash
# Install once as root to enable direnv for all interactive Bash users in this container.
# Each account must still review and allow the repository .envrc once (direnv security).
set -Eeuo pipefail
if (( EUID != 0 )); then
    echo '请执行: sudo bash scripts/install-global-direnv-hook.sh' >&2
    exit 1
fi
if ! command -v direnv >/dev/null 2>&1; then
    echo '请先安装: sudo apt-get install -y direnv' >&2
    exit 1
fi

cat > /etc/profile.d/90-direnv-shared.sh <<'HOOK'
# Managed by deepfake-env-kit; only interactive Bash, only once per shell.
if [ -n "${BASH_VERSION:-}" ]; then
    case "$-" in
        *i*)
            if [ -z "${_DEEPFAKE_DIRENV_HOOK_LOADED:-}" ] && command -v direnv >/dev/null 2>&1; then
                eval "$(direnv hook bash)"
                _DEEPFAKE_DIRENV_HOOK_LOADED=1
            fi
            ;;
    esac
fi
HOOK
chmod 644 /etc/profile.d/90-direnv-shared.sh

# Cover interactive non-login shells as well as login shells.
if ! grep -Fq '# deepfake-env-kit: global direnv Bash hook' /etc/bash.bashrc; then
    cat >> /etc/bash.bashrc <<'HOOK'

# deepfake-env-kit: global direnv Bash hook
if [ -r /etc/profile.d/90-direnv-shared.sh ]; then
    . /etc/profile.d/90-direnv-shared.sh
fi
HOOK
fi

echo '全局 direnv Bash hook 已安装。请重新打开 Bash 终端。'
echo '每位用户第一次进入此项目时仍需自行执行一次 direnv allow .'
