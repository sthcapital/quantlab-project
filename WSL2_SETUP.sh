# WSL2 Setup for QuantLab
# Run these commands in PowerShell (as Administrator) then in WSL2

# ════════════════════════════════════════════════════════════
# STEP 1 — Install WSL2 (run in PowerShell as Administrator)
# ════════════════════════════════════════════════════════════

# Install WSL2 with Ubuntu 22.04
wsl --install -d Ubuntu-22.04

# If WSL is already installed but needs upgrading:
# wsl --set-default-version 2
# wsl --install -d Ubuntu-22.04

# After install, reboot Windows when prompted.
# WSL2 will launch on next boot and ask you to create a Linux username/password.
# Use something simple — this is local only.


# ════════════════════════════════════════════════════════════
# STEP 2 — Inside WSL2 terminal: base Linux setup
# ════════════════════════════════════════════════════════════

sudo apt update && sudo apt upgrade -y
sudo apt install -y git curl wget build-essential python3.11 python3.11-venv python3-pip


# ════════════════════════════════════════════════════════════
# STEP 3 — Install Miniconda in WSL2
# (mirrors your Windows conda setup exactly)
# ════════════════════════════════════════════════════════════

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
bash ~/miniconda.sh -b -p ~/miniconda3
~/miniconda3/bin/conda init bash
source ~/.bashrc

# Verify
conda --version


# ════════════════════════════════════════════════════════════
# STEP 4 — Clone your repo into WSL2 (Linux filesystem)
# IMPORTANT: Work in ~/projects/ NOT in /mnt/c/
# Files on /mnt/c/ (Windows drive) are 5–10x slower in WSL2
# ════════════════════════════════════════════════════════════

mkdir -p ~/projects
cd ~/projects
git clone https://github.com/sthcapital/quantlab-project.git
cd quantlab-project


# ════════════════════════════════════════════════════════════
# STEP 5 — Create the conda environment in WSL2
# ════════════════════════════════════════════════════════════

conda create -n quantlab python=3.13 -y
conda activate quantlab

# Install all dependencies from pyproject.toml
pip install -e ".[dev]"

# Verify
python -c "import duckdb, pyarrow, scipy, sklearn; print('all deps OK')"
pytest -q


# ════════════════════════════════════════════════════════════
# STEP 6 — IBKR TWS note
# TWS runs on Windows. WSL2 can connect to it via the Windows host IP.
# Find your Windows host IP from inside WSL2:
# ════════════════════════════════════════════════════════════

# Run this inside WSL2 to get the Windows host IP:
cat /etc/resolv.conf | grep nameserver | awk '{print $2}'
# Example output: 172.20.0.1
# Use this IP instead of 127.0.0.1 when connecting from WSL2 to TWS:
# --host 172.20.0.1 --port 7497

# In TWS: File > Global Config > API > Settings
# Check "Allow connections from localhost only" → UNCHECK this
# Add the WSL2 IP to the trusted IPs list


# ════════════════════════════════════════════════════════════
# STEP 7 — Daily workflow in WSL2
# ════════════════════════════════════════════════════════════

# Open WSL2 terminal (search "Ubuntu" in Windows Start)
# OR: from PowerShell type: wsl
cd ~/projects/quantlab-project
conda activate quantlab

# Run scanner (use Windows host IP for IBKR)
WINDOWS_HOST=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}')
python scripts/scan_universe.py --universe small --signal breakout --host $WINDOWS_HOST

# Run backtest
python scripts/run_backtest.py --provider ibkr --symbol AAPL \
    --start 2025-01-01 --end 2026-06-03 \
    --signal breakout --lookback 20 \
    --host $WINDOWS_HOST --save-db

# Run tests
pytest -q


# ════════════════════════════════════════════════════════════
# STEP 8 — Optional: Set up cron for daily pre-market scan
# No PowerShell needed — standard Linux cron
# ════════════════════════════════════════════════════════════

# Edit crontab:
crontab -e

# Add this line (runs scan at 9:00 AM ET = 14:00 UTC):
# 0 14 * * 1-5 cd ~/projects/quantlab-project && /root/miniconda3/envs/quantlab/bin/python scripts/scan_universe.py --universe sp500_sample --no-news >> ~/quantlab-scan.log 2>&1

# View logs:
# tail -f ~/quantlab-scan.log


# ════════════════════════════════════════════════════════════
# STEP 9 — Configure PyCharm to use WSL2 interpreter
# (so you can edit in PyCharm and run in WSL2)
# ════════════════════════════════════════════════════════════

# In PyCharm:
# File > Settings > Python Interpreter > Add Interpreter
# Choose: "On WSL"
# Distribution: Ubuntu-22.04
# Path: /root/miniconda3/envs/quantlab/bin/python
# Click OK

# PyCharm will index the WSL2 interpreter automatically.
# You can now run/debug scripts directly in the WSL2 environment from PyCharm.
