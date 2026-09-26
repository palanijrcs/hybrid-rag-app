#!/usr/bin/env bash
# Set (or change) the admin password that unlocks Upload/Delete on the public site.
# On the VPS:  /opt/hybrid-rag-app/deploy/set-admin-password.sh
set -euo pipefail
cd "$(dirname "$0")/.."

read -rsp "New admin password (min 10 characters): " P1; echo
read -rsp "Type it again: " P2; echo
[ "$P1" = "$P2" ] || { echo "Passwords do not match."; exit 1; }
[ ${#P1} -ge 10 ] || { echo "Too short: use at least 10 characters."; exit 1; }
case "$P1" in *[\$\"\'\`\\]*) echo "Please avoid the characters \$ \" ' \` \\"; exit 1;; esac

sed -i '/^ADMIN_PASSWORD=/d' .env
printf 'ADMIN_PASSWORD=%s\n' "$P1" >> .env
chmod 600 .env
docker compose up -d frontend >/dev/null
echo "Admin password saved and the website restarted."
