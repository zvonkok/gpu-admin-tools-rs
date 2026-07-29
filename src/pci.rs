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

//! Sysfs PCI access: device identity, BAR0 mapping, function-level reset.
//! All paths are kernel contracts (`/sys/bus/pci/devices/<bdf>/...`), not
//! configuration.

use anyhow::{ensure, Context, Result};
use std::fs::OpenOptions;
use std::os::fd::AsRawFd;
use std::path::{Path, PathBuf};
use std::time::Duration;

pub const SYSFS_PCI: &str = "/sys/bus/pci/devices";

/// Read a sysfs attribute that prints as hex (`0x10de`).
pub fn attr_hex(dir: &Path, name: &str) -> Result<u32> {
    let text = std::fs::read_to_string(dir.join(name))
        .with_context(|| format!("read {}/{name}", dir.display()))?;
    let text = text.trim().trim_start_matches("0x");
    u32::from_str_radix(text, 16).with_context(|| format!("parse {}/{name}", dir.display()))
}

/// One PCI function with its BAR0 mapped for register access.
pub struct PciDev {
    pub bdf: String,
    pub vendor: u16,
    pub device: u16,
    path: PathBuf,
    bar0: Bar0,
    /// Previous `power/control` policy, restored on drop.
    power_control: Option<String>,
}

impl PciDev {
    /// Open `<SYSFS_PCI>/<bdf>` and map its BAR0 (`resource0`).  The domain
    /// may be omitted (`65:00.0` means `0000:65:00.0`).  Needs root.
    pub fn open(bdf: &str) -> Result<Self> {
        let bdf = if bdf.split(':').count() == 2 {
            format!("0000:{}", bdf.to_lowercase())
        } else {
            bdf.to_lowercase()
        };
        let path = Path::new(SYSFS_PCI).join(&bdf);
        ensure!(path.exists(), "no PCI device {bdf}");
        let power_control = wake(&path)?;
        Ok(Self {
            vendor: attr_hex(&path, "vendor")? as u16,
            device: attr_hex(&path, "device")? as u16,
            bar0: Bar0::map(&path.join("resource0"))?,
            bdf,
            path,
            power_control,
        })
    }

    pub fn read32(&self, offset: u32) -> u32 {
        self.bar0.read32(offset)
    }

    pub fn write32(&self, offset: u32, value: u32) {
        self.bar0.write32(offset, value)
    }

    /// Function-level reset via sysfs.  The kernel saves and restores config
    /// space around it, so the mapping stays valid.
    pub fn sysfs_reset(&self) -> Result<()> {
        std::fs::write(self.path.join("reset"), "1")
            .with_context(|| format!("{}: reset via sysfs", self.bdf))
    }
}

impl Drop for PciDev {
    fn drop(&mut self) {
        if let Some(previous) = &self.power_control {
            let _ = std::fs::write(self.path.join("power/control"), previous);
        }
    }
}

/// A runtime-suspended device (D3) reads all-ones on MMIO.  Force it to D0
/// by switching the runtime-PM policy to `on`; returns the previous policy
/// so drop can restore it.
fn wake(path: &Path) -> Result<Option<String>> {
    let control = path.join("power/control");
    let Ok(previous) = std::fs::read_to_string(&control) else {
        return Ok(None);
    };
    let previous = previous.trim().to_string();
    if std::fs::write(&control, "on").is_err() {
        // Not fatal here: without root the resource0 open below fails
        // with the clearer error.
        return Ok(None);
    }
    let status = path.join("power/runtime_status");
    crate::poll("device wakeup from D3", Duration::from_secs(5), || {
        std::fs::read_to_string(&status).is_ok_and(|s| s.trim() == "active")
    })?;
    Ok(Some(previous))
}

/// MMIO mapping of `resource0`.  Sysfs resource files reject read()/write();
/// mmap is the only access path.
struct Bar0 {
    ptr: *mut u8,
    len: usize,
}

// SAFETY: the pointer is a plain MMIO address; volatile accesses are not
// tied to the owning thread.  Deliberately not Sync — the FSP RPC
// sequences on top of this are not safe to interleave.
unsafe impl Send for Bar0 {}

impl Bar0 {
    fn map(resource0: &Path) -> Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(resource0)
            .with_context(|| format!("open {} (needs root)", resource0.display()))?;
        let len = file.metadata()?.len() as usize;
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                file.as_raw_fd(),
                0,
            )
        };
        ensure!(
            ptr != libc::MAP_FAILED,
            "mmap {}: {}",
            resource0.display(),
            std::io::Error::last_os_error()
        );
        Ok(Self {
            ptr: ptr as *mut u8,
            len,
        })
    }

    fn read32(&self, offset: u32) -> u32 {
        let offset = offset as usize;
        assert!(offset + 4 <= self.len, "BAR0 read past end: {offset:#x}");
        unsafe { (self.ptr.add(offset) as *const u32).read_volatile() }
    }

    fn write32(&self, offset: u32, value: u32) {
        let offset = offset as usize;
        assert!(offset + 4 <= self.len, "BAR0 write past end: {offset:#x}");
        unsafe { (self.ptr.add(offset) as *mut u32).write_volatile(value) }
    }
}

impl Drop for Bar0 {
    fn drop(&mut self) {
        unsafe { libc::munmap(self.ptr as *mut libc::c_void, self.len) };
    }
}
