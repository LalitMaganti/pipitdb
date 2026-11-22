#!/bin/sh
# Sums section sizes rather than file size, which includes page padding.
# .relro_padding is page padding too, as a section.
set -eu
target=$1

# On Linux, relative relocations are packed, as an embedder that can (glibc
# 2.36, Android API 28 and up) would: otherwise each pointer in a table of
# functions costs a 24-byte relocation, which hides what the code costs.
# Linked with the toolchain's lld, as GNU ld packs them only on x86_64.
case $target in
  *-linux-*)
    host=$(rustc -vV | awk '/^host:/ { print $2 }')
    lld="$(rustc --print sysroot)/lib/rustlib/$host/bin/gcc-ld"
    link="-C link-arg=-B$lld -C link-arg=-fuse-ld=lld -C link-arg=-Wl,-z,pack-relative-relocs"
    ;;
  *) link= ;;
esac
# shellcheck disable=SC2086 # $link is flags, split on purpose.
cargo rustc -q --release --target "$target" -p pipitdb-bench --example size --no-default-features --features size -- $link
case $target in
  wasm32-*) file=size.wasm ;;
  *-apple-*) file=libsize.dylib ;;
  *) file=libsize.so ;;
esac

llvm_size=$(find "$(rustc --print sysroot)" -name llvm-size -type f | head -n 1)
sections=$("$llvm_size" -A "target/$target/release/examples/$file")
size=$(echo "$sections" | awk 'NR > 2 && $1 != "Total" && $1 != ".relro_padding" { s += $2 } END { print s }')
budget=$(awk -v target="$target" '$1 == target { print $2 }' crates/bench/sizes)

echo "$target: $size bytes (budget: $budget)"
if [ "$size" -gt "$budget" ]; then
  echo "$sections"
  exit 1
fi
