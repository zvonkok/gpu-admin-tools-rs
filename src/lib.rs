// SPDX-FileCopyrightText: Copyright (c) 2018-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.

//! In-band NVIDIA GPU Confidential Computing (CC) control.
//!
//! Rust port of the CC subset of NVIDIA's gpu-admin-tools: query the
//! current CC mode, set a new one through FSP PRC knobs, and reset the
//! GPU so the new mode takes effect.  Talks to the GPU only through
//! sysfs (identity, reset) and a BAR0 mapping (registers) — no driver
//! involved.  The GPU must be idle: not held by nvidia/vfio while in use.
//!
//! ```no_run
//! use gpu_cc::{CcMode, Gpu};
//!
//! fn provision(bdf: &str) -> anyhow::Result<()> {
//!     let gpu = Gpu::open(bdf)?;
//!     if gpu.query_cc_mode()? != CcMode::On {
//!         gpu.set_cc_mode(CcMode::On)?;
//!         gpu.reset()?; // the new mode takes effect on reset
//!     }
//!     Ok(())
//! }
//! ```
//!
//! [`discover`] lists CC-capable GPUs without opening them (no root, no
//! device wake-up); [`Gpu::open`] maps BAR0 and needs root.

pub mod fsp;
pub mod pci;

use anyhow::{bail, ensure, Context, Result};
use fsp::FspRpc;
use pci::PciDev;
use std::time::{Duration, Instant};

/// The three CC modes (gpu-admin-tools `--set-cc-mode` values).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CcMode {
    /// Confidential computing disabled.
    Off,
    /// Full confidential computing.
    On,
    /// CC with the profiling/debugging interfaces left open.
    DevTools,
}

impl std::str::FromStr for CcMode {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        match s {
            "off" => Ok(Self::Off),
            "on" => Ok(Self::On),
            "devtools" => Ok(Self::DevTools),
            _ => bail!("invalid CC mode {s:?} (expected off, on or devtools)"),
        }
    }
}

impl std::fmt::Display for CcMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Off => "off",
            Self::On => "on",
            Self::DevTools => "devtools",
        })
    }
}

/// One CC-capable GPU generation: a PCI device-id range and the two
/// per-generation register facts.  Supporting a new chip is one row.
pub struct Chip {
    pub name: &'static str,
    /// Inclusive PCI device-id range.
    pub devid: (u16, u16),
    /// Hopper uses the EMEM RPC channel, extra PRC knobs and a different
    /// CC-state register; Blackwell uses MNOC and has a boot BAR0 firewall.
    pub hopper: bool,
    /// NV_THERM_I2CS_SCRATCH_FSP_BOOT_COMPLETE: reads 0xff once the FSP
    /// has finished booting the GPU.
    pub boot_complete: u32,
}

/// Device-id ranges from gpu-admin-tools (`gpu/devid_chips.py`).
#[rustfmt::skip]
pub const CHIPS: &[Chip] = &[
    Chip { name: "GH100", devid: (0x22f0, 0x237f), hopper: true, boot_complete: 0x200bc },
    Chip { name: "GB100", devid: (0x2900, 0x297f), hopper: false, boot_complete: 0x200bc },
    Chip { name: "GB102", devid: (0x2980, 0x29ff), hopper: false, boot_complete: 0x200bc },
    Chip { name: "GB110", devid: (0x3180, 0x31ff), hopper: false, boot_complete: 0x200bc },
    Chip { name: "GB112", devid: (0x3200, 0x327f), hopper: false, boot_complete: 0x200bc },
    Chip { name: "GB202", devid: (0x2b80, 0x2bff), hopper: false, boot_complete: 0xad00bc },
    Chip { name: "GB203", devid: (0x2c00, 0x2c7f), hopper: false, boot_complete: 0xad00bc },
    Chip { name: "GB205", devid: (0x2f00, 0x2f7f), hopper: false, boot_complete: 0xad00bc },
    Chip { name: "GB206", devid: (0x2d00, 0x2d7f), hopper: false, boot_complete: 0xad00bc },
    Chip { name: "GB207", devid: (0x2d80, 0x2dff), hopper: false, boot_complete: 0xad00bc },
];

/// C2C (Grace superchip) device ids: CC there is owned by system firmware,
/// so enabling it in-band is refused — disabling is still allowed
/// (gpu-admin-tools `has_c2c`).
const C2C_DEVIDS: &[u16] = &[
    0x2342, 0x2343, 0x2345, 0x2348, // GH200
    0x2941, 0x297e, 0x29bc, 0x31c2, // GB200/GB300
];

pub fn chip_for(devid: u16) -> Option<&'static Chip> {
    CHIPS
        .iter()
        .find(|c| (c.devid.0..=c.devid.1).contains(&devid))
}

const NV_PMC_BOOT_0: u32 = 0x0;
/// CC state lives in secure scratch, bits 1:0: 0 off, 1 on, 3 devtools.
const CC_STATE_HOPPER: u32 = 0x1182cc;
const CC_STATE_BLACKWELL: u32 = 0x590;
const BOOT_COMPLETE_OK: u32 = 0xff;

/// Poll until `done` returns true, at 1 ms granularity.
pub(crate) fn poll(what: &str, timeout: Duration, mut done: impl FnMut() -> bool) -> Result<()> {
    let start = Instant::now();
    loop {
        if done() {
            return Ok(());
        }
        if start.elapsed() > timeout {
            bail!("timed out waiting for {what}");
        }
        std::thread::sleep(Duration::from_millis(1));
    }
}

/// PCI addresses of all CC-capable NVIDIA GPUs on the node, sorted.
/// Reads only sysfs identity attributes: no root, no BAR0 mapping, and
/// no wake-up of runtime-suspended devices.
pub fn discover() -> Result<Vec<String>> {
    let mut bdfs = Vec::new();
    for entry in std::fs::read_dir(pci::SYSFS_PCI).context("read sysfs PCI tree")? {
        let entry = entry?;
        let Ok(bdf) = entry.file_name().into_string() else {
            continue;
        };
        let vendor = pci::attr_hex(&entry.path(), "vendor").unwrap_or(0);
        let device = pci::attr_hex(&entry.path(), "device").unwrap_or(0);
        if vendor == 0x10de && chip_for(device as u16).is_some() {
            bdfs.push(bdf);
        }
    }
    bdfs.sort();
    Ok(bdfs)
}

/// One CC-capable NVIDIA GPU.
pub struct Gpu {
    pci: PciDev,
    pub chip: &'static Chip,
    pub c2c: bool,
}

impl Gpu {
    /// Open a GPU by PCI address (`0000:65:00.0`; domain optional).
    pub fn open(bdf: &str) -> Result<Self> {
        let pci = PciDev::open(bdf)?;
        ensure!(
            pci.vendor == 0x10de,
            "{}: vendor {:#06x} is not NVIDIA",
            pci.bdf,
            pci.vendor
        );
        let chip = chip_for(pci.device).with_context(|| {
            format!(
                "{}: device {:#06x} is not a CC-capable GPU (Hopper or Blackwell)",
                pci.bdf, pci.device
            )
        })?;
        let c2c = C2C_DEVIDS.contains(&pci.device);
        let gpu = Self { pci, chip, c2c };

        gpu.wait_for_bar0()?;
        let boot0 = gpu.pci.read32(NV_PMC_BOOT_0);
        ensure!(boot0 != 0xffff_ffff, "{}: BAR0 not accessible", gpu.bdf());
        ensure!(
            boot0 != 0xbadf_0200 && boot0 != 0xbad0_0200,
            "{}: GPU is in a security-fault state (BOOT_0 = {boot0:#010x})",
            gpu.bdf()
        );
        Ok(gpu)
    }

    /// [`discover`] and open every CC-capable GPU.  Fails on the first GPU
    /// that cannot be opened; open individually for per-GPU error handling.
    pub fn enumerate() -> Result<Vec<Gpu>> {
        discover()?.iter().map(|bdf| Gpu::open(bdf)).collect()
    }

    pub fn bdf(&self) -> &str {
        &self.pci.bdf
    }

    pub fn devid(&self) -> u16 {
        self.pci.device
    }

    /// Blackwell keeps a BAR0 firewall up during boot; every register
    /// reads all-ones until the FSP lowers it.
    fn wait_for_bar0(&self) -> Result<()> {
        if self.chip.hopper {
            return Ok(());
        }
        poll("BAR0 firewall", Duration::from_secs(15), || {
            self.pci.read32(NV_PMC_BOOT_0) != 0xffff_ffff
        })
    }

    /// Wait until the FSP reports boot complete (scratch reads 0xff).
    pub fn wait_for_boot(&self) -> Result<()> {
        self.wait_for_bar0()?;
        poll("GPU boot complete", Duration::from_secs(10), || {
            self.pci.read32(self.chip.boot_complete) == BOOT_COMPLETE_OK
        })
    }

    /// The mode the GPU is currently running with.
    pub fn query_cc_mode(&self) -> Result<CcMode> {
        self.wait_for_boot()?;
        let reg = if self.chip.hopper {
            CC_STATE_HOPPER
        } else {
            CC_STATE_BLACKWELL
        };
        match self.pci.read32(reg) & 0x3 {
            0x0 => Ok(CcMode::Off),
            0x1 => Ok(CcMode::On),
            0x3 => Ok(CcMode::DevTools),
            _ => bail!(
                "{}: invalid CC state (devtools without CC); fix by setting a CC mode",
                self.bdf()
            ),
        }
    }

    /// Persist a new CC mode in the FSP.  It takes effect on the next GPU
    /// reset — call [`Gpu::reset`] afterwards.
    pub fn set_cc_mode(&self, mode: CcMode) -> Result<()> {
        if self.c2c && mode != CcMode::Off {
            bail!(
                "{}: enabling CC in-band is not supported on C2C (Grace) systems",
                self.bdf()
            );
        }
        // FSP RPC may come up before the boot-complete scratch; if it never
        // does, the RPC polls below fail with a clear error.
        let _ = self.wait_for_boot();

        let rpc = if self.chip.hopper {
            FspRpc::emem(&self.pci)?
        } else {
            FspRpc::mnoc(&self.pci)
        };

        let (ccm, ccd, bar0_decoupler) = match mode {
            CcMode::On => (1, 0, 2),
            CcMode::DevTools => (1, 1, 0),
            CcMode::Off => (0, 0, 0),
        };

        if self.chip.hopper {
            if ccm == 1 {
                // Knobs that conflict with CC are cleared first.
                for knob in [fsp::KNOB_2, fsp::KNOB_4, fsp::KNOB_34] {
                    rpc.knob_check_and_write(knob, 0)?;
                }
                match rpc.knob_read(fsp::KNOB_PPCIE) {
                    Ok(0) => {}
                    Ok(_) => rpc.knob_write(fsp::KNOB_PPCIE, 0)?,
                    // Older firmware without the PPCIE knob.
                    Err(err) if fsp::is_invalid_knob(&err) => {}
                    Err(err) => return Err(err),
                }
            }
            rpc.knob_check_and_write(fsp::KNOB_BAR0_DECOUPLER, bar0_decoupler)?;
        }

        // CCM goes on first and off last so a CCD-only state (invalid)
        // never exists.
        if ccm == 1 {
            rpc.knob_check_and_write(fsp::KNOB_CCM, ccm)?;
            rpc.knob_check_and_write(fsp::KNOB_CCD, ccd)?;
        } else {
            rpc.knob_check_and_write(fsp::KNOB_CCD, ccd)?;
            rpc.knob_check_and_write(fsp::KNOB_CCM, ccm)?;
        }
        Ok(())
    }

    /// Function-level reset, then wait for the GPU to boot back up.
    /// This is what makes a previously set CC mode active.
    pub fn reset(&self) -> Result<()> {
        self.pci.sysfs_reset()?;
        self.wait_for_boot()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chip_lookup() {
        assert_eq!(chip_for(0x2330).unwrap().name, "GH100"); // H100 SXM
        assert!(chip_for(0x2330).unwrap().hopper);
        assert_eq!(chip_for(0x2901).unwrap().name, "GB100");
        assert!(!chip_for(0x2901).unwrap().hopper);
        assert_eq!(chip_for(0x2b85).unwrap().name, "GB202");
        assert_eq!(chip_for(0x2b85).unwrap().boot_complete, 0xad00bc);
        assert!(chip_for(0x20b0).is_none()); // A100: no CC
    }

    #[test]
    fn c2c_blocks_enable_only() {
        assert!(C2C_DEVIDS.contains(&0x2342)); // GH200
        assert_eq!(chip_for(0x2342).unwrap().name, "GH100");
    }

    #[test]
    fn cc_mode_roundtrip() {
        for mode in [CcMode::Off, CcMode::On, CcMode::DevTools] {
            assert_eq!(mode.to_string().parse::<CcMode>().unwrap(), mode);
        }
        assert!("auto".parse::<CcMode>().is_err());
    }
}
