#!/bin/bash
# One command on Torch: update the repo, optionally save a new Drive link + token, submit the stage(s).
# Every job then uploads its out/<tag> folder to that Drive folder when it ends.
#   bash jobs/go.sh <stage> [stage...] [--drive <folder link> <token>]
# The token is written only to /scratch/$USER/.drive_upload/header (mode 600); it is never printed or committed.
set -e
R=/scratch/$USER/wsfinal-hpc
U=/scratch/$USER/.drive_upload
stages=()
while [ $# -gt 0 ]; do
  case "$1" in
    --drive) link=$2; token=$3; shift 3 ;;
    *) stages+=("$1"); shift ;;
  esac
done
if [ -n "$link" ]; then
  [ -n "$token" ] || { echo "usage: --drive <folder link> <token>"; exit 1; }
  id=$(echo "$link" | sed -nE 's#.*/folders/([A-Za-z0-9_-]+).*#\1#p; s#.*[?&]id=([A-Za-z0-9_-]+).*#\1#p' | head -1)
  [ -n "$id" ] || id=$link            # a bare folder id works too
  (umask 077; mkdir -p $U; printf '%s' "$id" > $U/folder_id; printf 'Authorization: Bearer %s' "$token" > $U/header)
  chmod 700 $U; chmod 600 $U/*
  unset token
  echo "Drive folder $id saved; checking the token with a small test file ..."
  echo "wsfinal upload check $(date -u '+%F %T UTC') from $(hostname)" > /scratch/$USER/wsfinal-upload-check.txt
  bash $R/jobs/upload.sh /scratch/$USER/wsfinal-upload-check.txt check
  rm -f /scratch/$USER/wsfinal-upload-check.txt
fi
[ ${#stages[@]} -gt 0 ] && bash $R/jobs/submit.sh "${stages[@]}"
true
