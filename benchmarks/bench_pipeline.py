"""Time a Diffusers pipeline with whole-model offload or partial offload.

    python benchmarks/bench_pipeline.py --model Efficient-Large-Model/Sana_1600M_1024px_diffusers \
        --mode partial --simulate-gib 8 --images 3 --out results.jsonl

``--mode``: ``model`` (``enable_model_cpu_offload``), ``group`` (``enable_group_offload`` with a stream, at
``--group-level``; ``block_level`` fails on Sana's DC-AE decoder in Diffusers 0.42.0.dev0), ``partial`` (this package),
``resident`` (everything on the GPU). ``--simulate-gib X`` first allocates a
blocking tensor so that only X GiB of device memory stay free (a smaller card on a bigger GPU). One untimed warm-up
image, then ``--images`` timed images with the same seed; prints and appends one JSON record with the seconds, the
NVML peak (with ``nvidia-ml-py``), the offload-extra report and the SHA-256 of the last image's pixels.

Every mode runs with ``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`` unless the variable is set, because
offload-extra turns expandable segments on by default. Set it to an empty string to test that runtime switch.
"""

import argparse
import hashlib
import json
import os
import threading
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np  # noqa: E402
import torch  # noqa: E402

GIB = 1 << 30


class NvmlPeak:
    def __init__(self):
        try:
            import pynvml

            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())
            self._nvml = pynvml
        except Exception:  # noqa: BLE001
            self._nvml = None
        self.peak, self._stop = 0, threading.Event()

    def __enter__(self):
        if self._nvml:
            threading.Thread(target=self._run, daemon=True).start()
        return self

    def _run(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, self._nvml.nvmlDeviceGetMemoryInfo(self._handle).used)
            time.sleep(0.005)

    def __exit__(self, *exc):
        self._stop.set()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", choices=["model", "group", "partial", "resident"], default="partial")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--variant")
    ap.add_argument("--prompt", default="A photo of a red fox sitting in fresh snow, morning light, shallow depth of field")
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--images", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--policy", default="belady")
    ap.add_argument("--host", default="keep")
    ap.add_argument("--cache-dir")
    ap.add_argument("--margin-gib", type=float, default=1.0)
    ap.add_argument("--host-embeddings", action="store_true")
    ap.add_argument("--group-level", choices=["leaf_level", "block_level"], default="leaf_level")
    ap.add_argument("--simulate-gib", type=float)
    ap.add_argument("--save")
    ap.add_argument("--out")
    args = ap.parse_args()

    from diffusers import DiffusionPipeline

    blocker = None
    if args.simulate_gib:
        free, _ = torch.cuda.mem_get_info()
        blocker = torch.empty(int(free - args.simulate_gib * GIB), dtype=torch.uint8, device="cuda")
    t0 = time.perf_counter()
    pipe = DiffusionPipeline.from_pretrained(args.model, torch_dtype=getattr(torch, args.dtype), variant=args.variant)
    offload = None
    if args.mode == "model":
        pipe.enable_model_cpu_offload()
    elif args.mode == "group":
        extra = {"num_blocks_per_group": 1} if args.group_level == "block_level" else {}
        pipe.enable_group_offload(onload_device=torch.device("cuda"), offload_type=args.group_level, use_stream=True,
                                  **extra)
    elif args.mode == "resident":
        pipe.to("cuda")
    else:
        import offload_extra

        offload = offload_extra.enable(pipe, "cuda", policy=args.policy, host=args.host, cache_dir=args.cache_dir,
                                         margin=int(args.margin_gib * GIB), host_embeddings=args.host_embeddings,
                                         timing=True)
    load_s = time.perf_counter() - t0

    def run():
        return pipe(args.prompt, height=args.height, width=args.width, num_inference_steps=args.steps,
                    generator=torch.Generator("cpu").manual_seed(args.seed)).images[0]

    with torch.inference_mode():
        t1 = time.perf_counter()
        run()
        warmup_s = time.perf_counter() - t1
        times, reports = [], []
        with NvmlPeak() as peak:
            for _ in range(args.images):
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                image = run()
                torch.cuda.synchronize()
                times.append(round(time.perf_counter() - t1, 3))
                if offload is not None:
                    reports.append(offload.last_call)
    if args.save:
        image.save(args.save)
    record = {
        "model": args.model, "mode": args.mode, "policy": args.policy if offload else None, "host": args.host,
        "group_level": args.group_level if args.mode == "group" else None,
        "size": f"{args.width}x{args.height}", "steps": args.steps, "simulate_gib": args.simulate_gib,
        "alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
        "gpu": torch.cuda.get_device_name(), "load_s": round(load_s, 1), "warmup_s": round(warmup_s, 2),
        "times_s": times, "best_s": min(times), "nvml_peak_gib": round(peak.peak / GIB, 2) if peak.peak else None,
        "pixels_sha256": hashlib.sha256(np.asarray(image).tobytes()).hexdigest()[:16],
        "partial": reports[-1] if reports else None,
        "units": offload.describe() if offload else None,
    }
    print(json.dumps(record))
    if args.out:
        with open(args.out, "a") as handle:
            handle.write(json.dumps(record) + "\n")
    del blocker


if __name__ == "__main__":
    main()
