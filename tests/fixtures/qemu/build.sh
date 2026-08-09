#!/usr/bin/env bash
# Build a tiny QEMU boot fixture (kernel + busybox initramfs) for the qemu
# backend integration test. Linux build host only. Pinned versions over
# HTTPS (add sha256 verification of the downloads before trusting this in CI).
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
work="$(mktemp -d)"; trap 'rm -rf "$work"' EXIT
cd "$work"

BUSYBOX_VER=1.36.1
LINUX_VER=6.6.30

# --- static busybox -------------------------------------------------------
curl -fsSLO "https://busybox.net/downloads/busybox-${BUSYBOX_VER}.tar.bz2"
tar xf "busybox-${BUSYBOX_VER}.tar.bz2"
cd "busybox-${BUSYBOX_VER}"
make defconfig
sed -i 's/# CONFIG_STATIC is not set/CONFIG_STATIC=y/' .config
# Fail loud if the sed no-oped (defconfig line format drift): a dynamically
# linked busybox in an initramfs with no /lib panics at boot instead.
grep -q '^CONFIG_STATIC=y' .config || { echo "ERROR: failed to enable CONFIG_STATIC in busybox .config" >&2; exit 1; }
make -j"$(nproc)" busybox
cd "$work"

# --- initramfs ------------------------------------------------------------
mkdir -p initramfs/{bin,proc,dev,root}
cp "busybox-${BUSYBOX_VER}/busybox" initramfs/bin/busybox
for a in sh cat id insmod mount uname stty; do ln -sf busybox "initramfs/bin/$a"; done
cat > initramfs/init <<'INIT'
#!/bin/sh
export PATH=/bin
/bin/busybox mkdir -p /proc /dev
/bin/busybox mount -t proc none /proc
/bin/busybox mount -t devtmpfs none /dev
# root-only flag; the challenge is to escalate via the kernel to read it
echo "CTF{fixture-has-no-flag}" > /root/flag
/bin/busybox chmod 600 /root/flag
# drop to an unprivileged shell on the console
exec /bin/busybox setuidgid 1000 /bin/sh
INIT
chmod +x initramfs/init
( cd initramfs && find . | cpio -o -H newc | gzip -9 ) > "$here/rootfs.cpio.gz"

# --- minimal kernel -------------------------------------------------------
curl -fsSLO "https://cdn.kernel.org/pub/linux/kernel/v6.x/linux-${LINUX_VER}.tar.xz"
tar xf "linux-${LINUX_VER}.tar.xz"
cd "linux-${LINUX_VER}"
make tinyconfig
# enable exactly what a serial-console initramfs boot needs
./scripts/config -e 64BIT -e TTY -e SERIAL_8250 -e SERIAL_8250_CONSOLE \
  -e PRINTK -e BINFMT_ELF -e BINFMT_SCRIPT -e BLK_DEV_INITRD -e DEVTMPFS \
  -e DEVTMPFS_MOUNT -e PROC_FS -e SYSFS -e MULTIUSER -e POSIX_TIMERS
make olddefconfig
make -j"$(nproc)" bzImage
cp arch/x86/boot/bzImage "$here/bzImage"
echo "built: $here/bzImage  $here/rootfs.cpio.gz"
