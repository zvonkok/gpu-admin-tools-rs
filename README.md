# gpu-cc

Query, set and reset NVIDIA GPU Confidential Computing (CC) mode, in Rust.
A port of the CC subset of [NVIDIA gpu-admin-tools] (MIT), which is the
reference implementation this crate mirrors register-for-register.

```sh
gpu-cc list                       # all CC-capable GPUs and their mode
gpu-cc query [BDF]                # off | on | devtools
gpu-cc set on [BDF] --reset       # switch mode, reset, verify
gpu-cc reset [BDF]                # sysfs function-level reset
```

## Library

The same package is a library (`gpu_cc`); the CLI is a thin wrapper over
it and is not built when the package is pulled in as a dependency:

```toml
[dependencies]
gpu-cc = { path = "../gpu-cc" }
# or, from another repo:
# gpu-cc = { git = "https://github.com/zvonkok/kata-device-plugin" }
```

```rust
use gpu_cc::{CcMode, Gpu};

// discover() lists CC-capable GPUs from sysfs identity alone —
// no root, no BAR0 mapping, no waking suspended devices.
for bdf in gpu_cc::discover()? {
    let gpu = Gpu::open(&bdf)?; // maps BAR0, needs root
    if gpu.query_cc_mode()? != CcMode::On {
        gpu.set_cc_mode(CcMode::On)?;
        gpu.reset()?; // the new mode takes effect on reset
    }
}
```

`Gpu` is `Send` (movable into a worker thread or blocking task) but not
`Sync`: the FSP RPC sequences must not be interleaved.

## How it works

Everything is in-band, driverless, and a kernel or hardware contract:

- **Query** — CC state is mirrored in a secure scratch register read
  straight from BAR0 (`0x1182cc` on Hopper, `0x590` on Blackwell,
  bits 1:0: `0` off, `1` on, `3` devtools).
- **Set** — CC configuration is persisted as PRC knobs owned by the FSP
  (firmware security processor).  Knobs are written with single-packet
  MCTP messages over a BAR0 mailbox: the FSP falcon EMEM channel 2 on
  Hopper, the FSP MNOC mailbox port 0 on Blackwell.
- **Reset** — the new mode only takes effect after a GPU reset:
  `echo 1 > /sys/bus/pci/devices/<bdf>/reset`, then wait for the
  BAR0 firewall (Blackwell) and the FSP boot-complete scratch.

BAR0 is mapped read-write from `/sys/bus/pci/devices/<bdf>/resource0`,
so the tool needs root, and the GPU must be idle — not in use by the
nvidia driver or a running VM.  A VFIO-bound idle GPU is typically
runtime-suspended (D3hot, MMIO reads all-ones), so the device is woken
via `power/control` first and the policy restored on exit.  Provisioning order on a Kata/CC node:
set CC mode (this tool) → bind to VFIO → pass through.

Supported chips are one table row each in `src/lib.rs` (`CHIPS`):
GH100 (H100/H800/H200), GB100/GB102/GB110/GB112 (B100/B200/B300),
GB202–GB207 (RTX Pro Blackwell).  On C2C Grace superchips (GH200/GB200)
CC is owned by system firmware, so enabling in-band is refused,
matching gpu-admin-tools.

## Build

```sh
cargo build --release
cargo test
cargo clippy -- -D warnings
```

[NVIDIA gpu-admin-tools]: https://github.com/NVIDIA/gpu-admin-tools
