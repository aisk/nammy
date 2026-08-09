# Performance

Reference numbers, and what had to be fixed to reach them. Nothing here is
needed to use nammy; see the [README](README.md) for that.

## Example training run

Reference numbers from runs on consumer hardware, training on the Blackstar
HT-1 capture pair from [Alec Wright's dataset](https://github.com/Alec-Wright/Automated-GuitarAmpModelling)
(340 s of aligned input/target at 44.1 kHz), on an **AMD Radeon RX 6800** on
Windows 11 via tinygrad's OpenCL (`CL`) backend, at the defaults otherwise:
batch 16, `ny` 8192, Adam(lr=0.004). For reference, an ESR below 0.01 is a good
model and 0.02–0.05 is usable.

**A2** (the default architecture) runs at **7.2 s per epoch**, after a first
epoch of 64 s, which puts 100 epochs at about **13 min**. Validation ESR is 0.39
after one epoch, 0.079 by epoch 3 and 0.058 by epoch 4. Unlike the A1 numbers
below, that is a per-epoch measurement from a short run rather than a completed
one, so the total is an extrapolation and there is no best-ESR figure to quote.
`ny` is tuned to 8299 automatically; the epoch is 30.8 s without that, which is
the subject of [Where the time goes](#where-the-time-goes).

**A1** (`--arch a1`) is a completed 100-epoch run: 61 s for the first epoch
(dominated by JIT compilation), then a steady ~8 s per epoch, 14 min total, for
a best validation ESR of **0.0070** reached at epoch 88. Convergence is fast:
ESR hits 0.054 by epoch 3 and 0.019 by epoch 10, then flattens while the
training loss keeps falling. The best checkpoint stands at 0.0086 by epoch 60
and 0.0079 by epoch 80, so the last 40 epochs are still worth about 20%.
Epoch-to-epoch validation noise is larger than that, though: ESR bounces between
0.0070 and 0.018 across epochs 40–100, so which epoch wins the best-checkpoint
pick is partly luck. At 8 s per epoch there is little reason to stop early.

## Where the time goes

The bottleneck is not raw compute. Both networks are narrow (16 and 8 channels
in A1, 8 in A2) but deep (20 and 23 layers), so every layer is a small kernel
that cannot fill a modern GPU, and the layers are serially dependent. A training
step at the default batch 16 / `ny` 8192 issues around 800 kernels for A1 and
950 for A2, and both sustain ~300–350 GFLOPS by tinygrad's op counter against
the RX 6800's ~16 TFLOPS FP32 peak, a few percent of the card.

Reaching even that took two fixes, both worth knowing about if you port this to
another backend:

- The backward pass reduces weight gradients over the (batch × time) axis,
  ~12k long against a handful of output elements. tinygrad only splits such a
  reduce across two kernels when the input/output element ratio reaches 32768,
  and this model sits just under that, so those reduces landed in single
  low-occupancy kernels: two of them alone cost 73 of the 188 ms step, running
  at 3–11 GFLOPS. `nammy/device.py` lowers `REDUCEOP_SPLIT_THRESHOLD` to 8192
  when it selects an accelerator, which takes the step to 63 ms; set it in the
  environment to override. A CPU has no occupancy to win back and measures a few
  percent slower, so the setting follows the device rather than being global.
- Validation ran one 65536-sample chunk at a time at batch 1, rebuilding the
  graph in Python for each. Chunks are independent, so `WaveNet.process` stacks
  them on the batch axis under a JIT: 5.3 s → 0.1 s per epoch.

Together those took the epoch from ~26 s to ~8 s, with the same results to
within float reordering. Memory is not a constraint: activations peak around
1 GB at the default batch and `ny`.

A2 hit a third, sharper version of the first problem. tinygrad splits a long
reduce by factoring the reduced axis, trying divisors from 256 down to 8, so an
axis whose factors all fall outside that window cannot be split at all, and
every sequence length in the network derives from the training window
`nx + ny - 1`. A2's default window is 6347 + 8192 - 1 = 14538 = 2·3·2423, which
has no such factor, so *none* of its weight-gradient reduces were split and one
kernel alone took 146 ms of the 293 ms step. Moving `ny` to 8299 makes the
window 14645 = 5·29·101 and the step 64 ms. `nammy/train.py:tune_ny` does this
automatically, searching within 1.5% of the `ny` it was given and leaving
windows that already factor alone (A1's 12284 is one, and the nearby windows
the search would otherwise prefer all measured slower). On the HT-1 pair that
is **30.8 s → 7.2 s per epoch**, 52 min → 13 min over 100 epochs.

What is left of the first epoch is tinygrad scheduling the graph in Python,
which `TinyJit` pays twice, once running eagerly and once capturing, for the
training step (~52 s) and again for validation's forward pass (~25 s). Compiling
the kernels themselves costs another ~2 minutes the first time a given window
length is seen on a machine, and no time afterwards: tinygrad keeps compiled
kernels in a sqlite cache that survives across runs.
