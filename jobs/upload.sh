#!/bin/bash
# Upload one folder to the Google Drive folder saved by go.sh, as an uncompressed tar.
#   bash jobs/upload.sh <folder> <name>          ->  <name>.tar in the Drive folder
# Credentials: /scratch/$USER/.drive_upload/{folder_id,header} (mode 600, written by go.sh, never in git or logs).
# A failed upload never fails the run: the files stay in /scratch/$USER/wsfinal/out/<tag>/.
U=/scratch/$USER/.drive_upload
src=$1; name=$2
[ -s $U/header ] && [ -s $U/folder_id ] || { echo "upload: no Drive token saved (run go.sh with the link and token)"; exit 0; }
FOLDER=$(cat $U/folder_id)
tmp=$(mktemp -d /scratch/$USER/upload.XXXXXX); trap 'rm -rf $tmp' EXIT
if [ -d "$src" ]; then tar cf $tmp/$name.tar -C "$(dirname "$src")" "$(basename "$src")"; f=$tmp/$name.tar; else f=$src; fi
size=$(stat -c %s "$f" 2>/dev/null || stat -f %z "$f")
for try in 1 2 3; do
  # Resumable upload: open a session (metadata + size), then PUT the bytes to the session URL.
  loc=$(curl -sS -D - -o $tmp/open.json -X POST -H @$U/header \
          -H "Content-Type: application/json; charset=UTF-8" -H "X-Upload-Content-Length: $size" \
          -d "{\"name\": \"$(basename "$f")\", \"parents\": [\"$FOLDER\"]}" \
          "https://www.googleapis.com/upload/drive/v3/files?uploadType=resumable&supportsAllDrives=true" \
        | tr -d '\r' | awk 'tolower($1) == "location:" {print $2}')
  if [ -z "$loc" ]; then
    echo "upload: Drive refused the session (try $try): $(head -c 300 $tmp/open.json | tr '\n' ' ')"
    grep -q '"code": 401' $tmp/open.json && { echo "upload: token expired or wrong: run go.sh with a new token"; exit 0; }
  else
    code=$(curl -sS -o $tmp/done.json -w '%{http_code}' -X PUT --upload-file "$f" "$loc")
    [ "$code" = 200 ] || [ "$code" = 201 ] && { echo "upload: $(basename "$f") ($((size / 1048576)) MB) -> Drive folder"; exit 0; }
    echo "upload: HTTP $code (try $try): $(head -c 300 $tmp/done.json | tr '\n' ' ')"
  fi
  sleep $((30 * try))
done
echo "upload: FAILED after 3 tries; the files are still in $src"
exit 0
