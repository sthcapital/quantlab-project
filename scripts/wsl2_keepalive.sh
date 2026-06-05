#!/usr/bin/env bash
# =============================================================================
# scripts/wsl2_keepalive.sh — Ensure the cron service is running.
#
# With systemd=true in /etc/wsl.conf, cron is managed by systemd and starts
# automatically when WSL2 boots.  This script is a safety net for edge cases
# where the service has been stopped or failed.
#
# Run once to install into /etc/profile.d/ so it executes on every login shell:
#   bash scripts/wsl2_keepalive.sh --install
#
# After installation, every new terminal will silently verify cron is active.
# =============================================================================

set -euo pipefail

PROFILE_D_DEST="/etc/profile.d/quantlab-keepalive.sh"
SELF="$(realpath "${BASH_SOURCE[0]}")"

# ── Install mode ──────────────────────────────────────────────────────────────
if [[ "${1:-}" == "--install" ]]; then
    echo "Installing keepalive to $PROFILE_D_DEST ..."

    if ! command -v sudo &>/dev/null; then
        echo "ERROR: sudo not found — install manually: cp $SELF $PROFILE_D_DEST"
        exit 1
    fi

    sudo cp "$SELF" "$PROFILE_D_DEST"
    sudo chmod 755 "$PROFILE_D_DEST"
    echo "[OK] Installed: $PROFILE_D_DEST"
    echo ""
    echo "Cron will be verified on every new login shell."
    echo "To test: open a new terminal and run: systemctl is-active cron"
    exit 0
fi

# ── Keepalive logic (runs silently from /etc/profile.d/) ─────────────────────
# Use non-interactive sudo (-n) so there is no password prompt;
# if cron is already running (the normal case), nothing happens.

_cron_running() {
    if command -v systemctl &>/dev/null; then
        systemctl is-active --quiet cron 2>/dev/null
    else
        service cron status &>/dev/null
    fi
}

if ! _cron_running; then
    if command -v systemctl &>/dev/null; then
        sudo -n systemctl start cron 2>/dev/null || true
    else
        sudo -n service cron start 2>/dev/null || true
    fi
fi
