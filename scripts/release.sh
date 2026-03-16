#!/usr/bin/env bash
# Create a GitHub release for the given version: tag, push, and use the
# corresponding section from CHANGELOG.md as release notes.
# Requires: gh (GitHub CLI), jq
# Usage: scripts/release.sh [--dry-run] <version>
# Example: scripts/release.sh 0.1.0
# Example: scripts/release.sh --dry-run 0.1.0

set -e

DRY_RUN=false
if [[ "$1" == "--dry-run" ]]; then
  DRY_RUN=true
  shift
fi
V="$1"
if [[ -z "$V" ]]; then
  echo "Usage: $0 [--dry-run] <version>" >&2
  echo "Example: $0 0.1.0" >&2
  echo "Example: $0 --dry-run 0.1.0" >&2
  exit 1
fi
if [[ "$2" == "--dry-run" ]]; then
  DRY_RUN=true
fi

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

# Ensure version matches Cargo.toml and package.json (same checks as publish workflow)
cargo_version="$(grep '^version = ' Cargo.toml | sed -n 's/^version = "\(.*\)"$/\1/p')"
if [[ "$cargo_version" != "$V" ]]; then
  echo "Error: Cargo.toml version is $cargo_version, expected $V" >&2
  exit 1
fi

pkg_version="$(jq -r '.version' npm/kriging-rs-wasm/package.json)"
if [[ "$pkg_version" != "$V" ]]; then
  echo "Error: npm/kriging-rs-wasm/package.json version is $pkg_version, expected $V" >&2
  exit 1
fi

# Extract this version's section from CHANGELOG.md (Keep a Changelog format)
notes_file="$(mktemp)"
trap 'rm -f "$notes_file"' EXIT
awk -v v="$V" '
  $0 ~ "^## \\[" v "\\]" { found=1; print; next }
  found && $0 ~ "^## \\[" { found=0; next }
  found && $0 ~ "^\\[" v "\\]:" { found=0; next }
  found { print }
' CHANGELOG.md > "$notes_file"

if [[ ! -s "$notes_file" ]]; then
  echo "Error: No section for [$V] in CHANGELOG.md" >&2
  exit 1
fi

TAG="v$V"
echo "Tag that will be created: $TAG"
if [[ "$DRY_RUN" == true ]]; then
  echo "[DRY RUN] Would create release $TAG with the following notes from CHANGELOG.md:"
  echo "---"
  cat "$notes_file"
  echo "---"
  echo "[DRY RUN] Skipping gh release create."
  exit 0
fi
echo "Creating release $TAG with notes from CHANGELOG.md..."
gh release create "$TAG" --notes-file "$notes_file"
echo "Done."
