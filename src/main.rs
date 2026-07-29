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

use anyhow::{bail, ensure, Result};
use gpu_cc::{CcMode, Gpu};

const USAGE: &str = "\
gpu-cc — query, set and reset NVIDIA GPU Confidential Computing (CC) mode

USAGE:
    gpu-cc list
    gpu-cc query [BDF]
    gpu-cc set <off|on|devtools> [BDF] [--reset]
    gpu-cc reset [BDF]

BDF is a PCI address like 0000:65:00.0 (domain optional).  It may be
omitted when the node has exactly one CC-capable GPU.

A new CC mode takes effect on the next GPU reset; `set --reset` resets
immediately and verifies the switch.  Needs root; the GPU must be idle.";

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut args: Vec<&str> = args.iter().map(String::as_str).collect();
    let reset_after = {
        let before = args.len();
        args.retain(|a| *a != "--reset");
        before != args.len()
    };

    match args.as_slice() {
        ["list"] => {
            for bdf in gpu_cc::discover()? {
                match Gpu::open(&bdf) {
                    Ok(gpu) => {
                        let mode = gpu
                            .query_cc_mode()
                            .map_or_else(|e| e.to_string(), |m| m.to_string());
                        println!(
                            "{}  {}  devid {:#06x}  cc {}",
                            gpu.bdf(),
                            gpu.chip.name,
                            gpu.devid(),
                            mode
                        );
                    }
                    Err(err) => println!("{bdf}  error: {err:#}"),
                }
            }
        }
        ["query", rest @ ..] => {
            let gpu = pick_gpu(rest)?;
            println!("{}", gpu.query_cc_mode()?);
        }
        ["set", mode, rest @ ..] => {
            let mode: CcMode = mode.parse()?;
            let gpu = pick_gpu(rest)?;
            gpu.set_cc_mode(mode)?;
            if reset_after {
                gpu.reset()?;
                let now = gpu.query_cc_mode()?;
                ensure!(
                    now == mode,
                    "{}: CC mode is {now} after reset, expected {mode}",
                    gpu.bdf()
                );
                println!("{}: CC mode {mode} is active", gpu.bdf());
            } else {
                println!(
                    "{}: CC mode set to {mode}; it takes effect on the next GPU reset",
                    gpu.bdf()
                );
            }
        }
        ["reset", rest @ ..] => {
            let gpu = pick_gpu(rest)?;
            gpu.reset()?;
            println!("{}: reset done", gpu.bdf());
        }
        _ => {
            eprintln!("{USAGE}");
            std::process::exit(2);
        }
    }
    Ok(())
}

fn pick_gpu(rest: &[&str]) -> Result<Gpu> {
    match rest {
        [bdf] => Gpu::open(bdf),
        [] => {
            let bdfs = gpu_cc::discover()?;
            match bdfs.as_slice() {
                [] => bail!("no CC-capable NVIDIA GPU found"),
                [bdf] => Gpu::open(bdf),
                _ => bail!(
                    "{} CC-capable GPUs found ({}); pass a BDF",
                    bdfs.len(),
                    bdfs.join(", ")
                ),
            }
        }
        _ => bail!("unexpected arguments: {rest:?}\n\n{USAGE}"),
    }
}
