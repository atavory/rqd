# Remote GPU Bundle - WSDM Semantic-ID Runs

This bundle is self-contained for the next GPU machine. It includes:

- `rqd/`: public experiment runner code;
- `cache/`: prepared `.npz` caches for ML-1M, Amazon2018 Electronics, and
  Amazon2023 Beauty/Tools/Toys;
- `DACT/`: cloned external DACT repo for protocol triage and competitor work;
- `run_remote_gpu_queue_20260814.sh`: restart-safe GPU queue for the immediate
  WSDM blocker runs;
- `remote_status_20260814.sh`: compact status checker.

## Setup On The Remote GPU Host

```bash
tar --zstd -xf wsdm_remote_gpu_bundle_20260814.tar.zst
cd wsdm_remote_gpu_bundle_20260814

python3 -m venv venv
./venv/bin/pip install --upgrade pip
./venv/bin/pip install -r rqd/requirements.txt
```

If the remote host already has a working PyTorch CUDA environment, set `PY` to
that interpreter instead of creating a new venv.

## Launch

```bash
PY="$PWD/venv/bin/python" \
OUT_ROOT="$PWD/results_remote_gpu_20260814" \
./run_remote_gpu_queue_20260814.sh
```

The queue runs Amazon2023 Beauty/Tools/Toys downstream rung matrices using the
prepared caches. This is the immediate Pareto/retrain-decision evidence:
frozen, stratified, GRM-only, warm/full update, EMA, relabel controls, and
rebuilt/full migration rows are produced by `run_wsdm_web_recsys.py`.

Defaults:

- categories: `beauty tools toys`;
- arch: `funnel24`;
- freeze depths: `1 2 3`;
- seeds: `0 1 2`;
- epochs: `50`;
- two GPU workers using `CUDA_VISIBLE_DEVICES=0` and `1`.

Override as needed:

```bash
SEEDS="0 1 2 3 4" ARCHES="funnel24 balanced24 uniform24" ./run_remote_gpu_queue_20260814.sh
```

## Status

```bash
OUT_ROOT="$PWD/results_remote_gpu_20260814" ./remote_status_20260814.sh
```

## What This Does And Does Not Cover

This queue covers the local strategy frontier and retrain-decision matrix on
Beauty/Tools/Toys. It is **not** yet the final external DACT/Reformer
comparison. The DACT clone is included so that external-protocol triage can be
done on the same machine, but exact DACT/Reformer rows still need explicit
integration or a documented substitute.

Interpret results as Pareto/frontier and rung-selection evidence, not as a
"better ranking without retraining" claim.

