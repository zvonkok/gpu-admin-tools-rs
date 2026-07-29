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

//! FSP PRC-knob RPC — the in-band interface that owns CC configuration.
//!
//! The FSP (firmware security processor) stores CC settings as persistent
//! "PRC knobs".  Knobs are read and written with single-packet MCTP
//! messages (NVIDIA vendor-defined type) delivered through one of two
//! BAR0 mailboxes:
//!
//! * Hopper — the FSP falcon's EMEM shared memory, channel 2
//! * Blackwell — the FSP MNOC mailbox, port 0
//!
//! Register offsets, framing and command values mirror NVIDIA's
//! gpu-admin-tools (MIT), the reference implementation for out-of-driver
//! CC provisioning.

use crate::pci::PciDev;
use crate::poll;
use anyhow::{bail, ensure, Result};
use std::time::Duration;

/// PRC knob ids (gpu-admin-tools `gpu/prc.py`).
pub const KNOB_2: u32 = 2; // Hopper-only, cleared before enabling CC
pub const KNOB_4: u32 = 4; // Hopper-only, cleared before enabling CC
pub const KNOB_CCD: u32 = 6; // CC devtools mode
pub const KNOB_CCM: u32 = 8; // CC mode
pub const KNOB_BAR0_DECOUPLER: u32 = 10; // Hopper BAR0 filter
pub const KNOB_34: u32 = 34; // Hopper-only, cleared before enabling CC
pub const KNOB_PPCIE: u32 = 45; // protected PCIe, mutually exclusive with CC

/// NVDM (NVIDIA data model) message types.
const NVDM_PRC: u32 = 0x13;
const NVDM_RESPONSE: u32 = 0x15;

/// A non-zero FSP completion code.
#[derive(Debug)]
pub struct FspError {
    pub code: u32,
}

impl std::fmt::Display for FspError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "FSP RPC failed with completion code {:#x}", self.code)
    }
}

impl std::error::Error for FspError {}

/// Firmware predating a knob answers knob reads with this code.
pub fn is_invalid_knob(err: &anyhow::Error) -> bool {
    err.downcast_ref::<FspError>()
        .is_some_and(|fsp| fsp.code == 0x1e3)
}

/// MCTP transport header: version 0, endpoint ids 0, tag 0, seq 0,
/// som=1 (bit 31), eom=1 (bit 30) — every RPC here fits one packet.
const MCTP_HEADER: u32 = 0xc000_0000;

/// MCTP message header dword: type 0x7e (vendor via PCI), vendor 0x10de,
/// NVDM type in the top byte.
fn mctp_msg_header(nvdm_type: u32) -> u32 {
    0x0010_de7e | nvdm_type << 24
}

fn mctp_packet(nvdm_type: u32, payload: &[u32]) -> Vec<u32> {
    let mut packet = Vec::with_capacity(2 + payload.len());
    packet.push(MCTP_HEADER);
    packet.push(mctp_msg_header(nvdm_type));
    packet.extend_from_slice(payload);
    packet
}

/// One RPC channel to the FSP.  Variant selection is per GPU generation.
pub enum FspRpc<'a> {
    Emem(&'a PciDev),
    Mnoc(&'a PciDev),
}

impl<'a> FspRpc<'a> {
    /// Hopper: EMEM channel 2.  Resets stale queue state left behind by a
    /// crashed client before first use.
    pub fn emem(dev: &'a PciDev) -> Result<Self> {
        emem_reset_queues(dev)?;
        Ok(Self::Emem(dev))
    }

    /// Blackwell: MNOC port 0.
    pub fn mnoc(dev: &'a PciDev) -> Self {
        Self::Mnoc(dev)
    }

    pub fn knob_read(&self, knob: u32) -> Result<u16> {
        let data = self.prc_cmd(&[0xc | 0x2 << 8 | knob << 16])?;
        ensure!(data.len() == 1, "knob {knob} read: bad response {data:x?}");
        // The value is 16 bits; the upper half is not zero-initialized.
        Ok(data[0] as u16)
    }

    pub fn knob_write(&self, knob: u32, value: u16) -> Result<()> {
        let data = self.prc_cmd(&[0xd | 0x2 << 8 | knob << 16, value as u32])?;
        ensure!(data.is_empty(), "knob {knob} write: bad response {data:x?}");
        Ok(())
    }

    pub fn knob_check_and_write(&self, knob: u32, value: u16) -> Result<()> {
        if self.knob_read(knob)? != value {
            self.knob_write(knob, value)?;
        }
        Ok(())
    }

    fn prc_cmd(&self, payload: &[u32]) -> Result<Vec<u32>> {
        let packet = mctp_packet(NVDM_PRC, payload);
        let response = match self {
            Self::Emem(dev) => {
                emem_send(dev, &packet)?;
                emem_receive(dev)?
            }
            Self::Mnoc(dev) => {
                mnoc_send(dev, &packet)?;
                mnoc_receive(dev)?
            }
        };
        // [mctp hdr, msg hdr, seq, request nvdm type, completion, payload...]
        ensure!(response.len() >= 5, "FSP response too short: {response:x?}");
        ensure!(
            response[1] >> 24 == NVDM_RESPONSE,
            "FSP response has wrong nvdm type: {response:x?}"
        );
        ensure!(
            response[3] == NVDM_PRC,
            "FSP response for wrong command: {response:x?}"
        );
        if response[4] != 0 {
            return Err(FspError { code: response[4] }.into());
        }
        Ok(response[5..].to_vec())
    }
}

// --- Hopper: FSP falcon EMEM, channel 2 ------------------------------------
//
// A command is written into the channel's EMEM window and announced by the
// queue head/tail registers; the response arrives in the same window,
// announced by the message queue registers.

const EMEM_CHANNEL: u32 = 2;
const EMEM_BASE: u32 = EMEM_CHANNEL * 1024; // byte offset inside EMEM
const EMEMC: u32 = 0x8f2ac0 + EMEM_CHANNEL * 8; // port control: offset + autoinc
const EMEMD: u32 = EMEMC + 4; // port data window
const QUEUE_HEAD: u32 = 0x8f2c00 + EMEM_CHANNEL * 8; // writing head is the doorbell
const QUEUE_TAIL: u32 = QUEUE_HEAD + 4;
const MSGQ_HEAD: u32 = 0x8f2c80 + EMEM_CHANNEL * 8;
const MSGQ_TAIL: u32 = MSGQ_HEAD + 4;
const EMEMC_AINCW: u32 = 1 << 24;
const EMEMC_AINCR: u32 = 1 << 25;

fn emem_reset_queues(dev: &PciDev) -> Result<()> {
    let empty = |head, tail| dev.read32(head) == dev.read32(tail);
    if empty(QUEUE_HEAD, QUEUE_TAIL) && empty(MSGQ_HEAD, MSGQ_TAIL) {
        return Ok(());
    }
    // Give an in-flight command a chance to produce its response, then
    // point both queues back at this channel's EMEM base.
    let _ = poll("stale FSP response", Duration::from_secs(5), || {
        !empty(MSGQ_HEAD, MSGQ_TAIL)
    });
    dev.write32(QUEUE_TAIL, EMEM_BASE);
    dev.write32(QUEUE_HEAD, EMEM_BASE);
    dev.write32(MSGQ_TAIL, EMEM_BASE);
    dev.write32(MSGQ_HEAD, EMEM_BASE);
    Ok(())
}

fn emem_send(dev: &PciDev, data: &[u32]) -> Result<()> {
    ensure!(data.len() * 4 <= 1024, "FSP command exceeds EMEM channel");
    poll("FSP command queue empty", Duration::from_secs(5), || {
        dev.read32(QUEUE_HEAD) == dev.read32(QUEUE_TAIL)
    })?;
    dev.write32(EMEMC, EMEM_BASE | EMEMC_AINCW | EMEMC_AINCR);
    for &d in data {
        dev.write32(EMEMD, d);
    }
    dev.write32(QUEUE_TAIL, EMEM_BASE + (data.len() as u32 - 1) * 4);
    dev.write32(QUEUE_HEAD, EMEM_BASE);
    Ok(())
}

fn emem_receive(dev: &PciDev) -> Result<Vec<u32>> {
    poll("FSP response", Duration::from_secs(5), || {
        dev.read32(MSGQ_HEAD) != dev.read32(MSGQ_TAIL)
    })?;
    let head = dev.read32(MSGQ_HEAD);
    let tail = dev.read32(MSGQ_TAIL);
    let dwords = tail.wrapping_sub(head) / 4 + 1;
    ensure!(dwords <= 256, "FSP response exceeds EMEM channel");
    dev.write32(EMEMC, EMEM_BASE | EMEMC_AINCW | EMEMC_AINCR);
    let data = (0..dwords).map(|_| dev.read32(EMEMD)).collect();
    dev.write32(MSGQ_TAIL, head); // ack
    Ok(data)
}

// --- Blackwell: FSP MNOC mailbox, port 0 ------------------------------------
//
// Two mailbox register pairs: we push commands through the "receive"
// mailbox (info + data) and pull responses from the "send" mailbox.

const MNOC_INFO_SEND: u32 = 0x8f1e00 + 0x104;
const MNOC_RDATA_SEND: u32 = MNOC_INFO_SEND + 4;
const MNOC_INFO_RECV: u32 = 0x8f1e00 + 0x184;
const MNOC_WDATA_RECV: u32 = MNOC_INFO_RECV + 4;
const MNOC_SIZE_MASK: u32 = 0xfffff;
const MNOC_NEW_MSG: u32 = 1 << 20;
const MNOC_READY: u32 = 1 << 24;
const MNOC_ERROR: u32 = 1 << 25;
const MNOC_CREDITS: u32 = 1 << 26;

fn mnoc_send(dev: &PciDev, data: &[u32]) -> Result<()> {
    poll("FSP MNOC receive ready", Duration::from_secs(5), || {
        dev.read32(MNOC_INFO_RECV) & MNOC_READY != 0
    })?;
    dev.write32(MNOC_INFO_RECV, (data.len() as u32 * 4) | MNOC_NEW_MSG);
    for (i, &d) in data.iter().enumerate() {
        // Credits are granted in 64-byte units.
        if i % 16 == 0 {
            poll("FSP MNOC credits", Duration::from_secs(1), || {
                dev.read32(MNOC_INFO_RECV) & MNOC_CREDITS != 0
            })?;
        }
        dev.write32(MNOC_WDATA_RECV, d);
    }
    let info = dev.read32(MNOC_INFO_RECV);
    ensure!(info & MNOC_ERROR == 0, "FSP MNOC send error: {info:#x}");
    Ok(())
}

fn mnoc_receive(dev: &PciDev) -> Result<Vec<u32>> {
    if let Err(err) = poll("FSP MNOC response", Duration::from_secs(5), || {
        dev.read32(MNOC_INFO_SEND) & MNOC_READY != 0
    }) {
        let info = dev.read32(MNOC_INFO_SEND);
        if info & MNOC_ERROR != 0 {
            bail!("FSP MNOC receive error: {info:#x}");
        }
        return Err(err);
    }
    let bytes = dev.read32(MNOC_INFO_SEND) & MNOC_SIZE_MASK;
    let data = (0..bytes / 4)
        .map(|_| dev.read32(MNOC_RDATA_SEND))
        .collect();
    let info = dev.read32(MNOC_INFO_SEND);
    ensure!(info & MNOC_ERROR == 0, "FSP MNOC receive error: {info:#x}");
    Ok(data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn knob_read_packet_matches_reference() {
        // gpu-admin-tools framing for a CCM knob read over PRC (nvdm 0x13).
        let payload = 0xc | 0x2 << 8 | KNOB_CCM << 16;
        assert_eq!(
            mctp_packet(NVDM_PRC, &[payload]),
            vec![0xc000_0000, 0x1310_de7e, 0x0008_020c]
        );
    }

    #[test]
    fn knob_write_packet_matches_reference() {
        let payload = 0xd | 0x2 << 8 | KNOB_CCD << 16;
        assert_eq!(
            mctp_packet(NVDM_PRC, &[payload, 0x1]),
            vec![0xc000_0000, 0x1310_de7e, 0x0006_020d, 0x1]
        );
    }
}
