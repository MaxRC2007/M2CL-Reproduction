# START HERE: round 4 (the authors' own code) on the institute Linux machine

This brief is for the person running the machine **and for any AI coding assistant helping them** (Claude in Antigravity, or similar). Read all of it before running anything.

## What this is
Rounds 1-3 tested the contrastive term of M2-CL (Ballas & Diou, IEEE TAI 2024) in our own reimplementation. **Round 4 runs the authors' released code itself** on PACS, with the contrastive weight `lparam` at 0.01 and at 0, paired by seed: 24 runs (2 settings x 4 test domains x 3 seeds). The plan was fixed and pushed to GitHub before any run (`docs/PREREGISTRATION_ROUND4.md`); read it.

The machine: NVIDIA RTX 4000 Ada (20 GB), Intel Core Ultra 9, Linux. **Start only after Part C (VLCS) has finished** (it printed `Done.`), so the two never share the GPU.

## Rules (they protect the pre-registration)
1. **Don't edit** anything in `configs/`, `src/`, `scripts/`, or the authors' code that the script downloads (`~/data/authors_m2cl_0b1bce3`). The script checks that the authors' code is the unmodified commit and refuses to run otherwise.
2. **Use only these options** of `run_round4_authors.py`: `--out`, `--data_dir`, `--parallel`, `--smoke`. Never pass `--no_gpu_check`.
3. **Don't delete or edit** anything in `~/m2cl_round4`. To continue after any stop, run the same command again. The authors' code cannot resume a run, so the script deletes an *unfinished* run and restarts it with the same seeds. Finished runs are always kept.
4. **If something fails, stop and report.** Collect the last 30 lines of the run's log and send them to the project owner. Don't "fix" it by changing code or settings. The only fixes allowed without the owner are installing the packages listed below.
5. Don't install or upgrade anything beyond the commands below. Don't change the GPU driver.

## Steps
```bash
# 0. Unzip the package into ~/m2cl_r4 (a new folder; the Part C folder ~/m2cl stays as it is)
mkdir -p ~/m2cl_r4 && cd ~/m2cl_r4 && unzip ~/Downloads/M2CL_Round4_institute.zip
# 1. Reuse the Python environment from Part C, then add the two packages the authors' code imports
conda activate m2cl            # or: source ~/m2cl-env/bin/activate
pip install -r requirements_round4.txt
python -c "import torch, wilds, timm; print(torch.__version__, torch.cuda.get_device_name(0))"
# 2. Optional, quick: the tests (about 5-10 minutes; all must pass)
python -m pytest -q tests/test_round4.py
# 3. In tmux (tmux new -s round4; detach Ctrl+B then D; return: tmux attach -t round4)
python scripts/run_round4_authors.py --out ~/m2cl_round4 --data_dir ~/data --smoke
#    downloads the authors' code (pinned commit) and DomainBed's PACS (0.18 GB, checks the 9,991 images),
#    trains 101 steps, then prints the time per step, the peak GPU memory and an estimate for all 24 runs.
# 4. All 24 runs. Use --parallel 2 if twice the smoke run's peak GPU memory fits well within 20 GB; else 1.
python scripts/run_round4_authors.py --out ~/m2cl_round4 --data_dir ~/data --parallel 2
# 5. When it prints "Done.":
cd ~ && zip -r m2cl_round4.zip m2cl_round4
```
Upload `m2cl_round4.zip` to Google Drive and download it to the laptop's `Downloads` folder. It holds every run's `results.jsonl`, final model, logs and the hardware record (`env/`).

## How to check progress
- The tmux window shows `[start]` and `[done] <run>: <accuracy>% ...` lines.
- `ls ~/m2cl_round4/round4/*/`: a run is finished when its folder has a file called `done`.
- `tail -f ~/m2cl_round4/logs/<run>.log` follows one run.
- `nvidia-smi` shows GPU use.

## Known problems and what to do
| Symptom | Cause | What to do |
|---|---|---|
| `The authors' code imports ['wilds', ...]` | Packages missing | `pip install -r requirements_round4.txt` |
| `ModuleNotFoundError` for another package, inside a run log | The authors' code imports something more | Stop and report the line. Installing that one package is allowed; record its name and version for the owner |
| `CUDA out of memory` | Two runs don't fit | Ctrl+C, then restart with `--parallel 1`; finished runs are kept |
| `Bus error` / `DataLoader worker ... killed` | Too little shared memory (`/dev/shm`); the authors' code uses 8 loader workers | Ask the admin to raise shared memory; don't change the code |
| `not the pre-registered unmodified commit` | The authors' folder was changed | Delete `~/data/authors_m2cl_0b1bce3` and run again (it is downloaded fresh) |
| `wrong image counts` | Bad PACS download | Delete `~/data/domainbed_pacs.partial` (only that) and run again |
| A download fails (GitHub, Hugging Face or download.pytorch.org) | Network or firewall | Check the internet connection, then run again |
| Any other traceback | Unknown | Stop. Send the last 30 lines of the run's log to the owner; don't edit code |

## What the AI assistant may and may not do here
- **May:** run the commands above; read logs and explain errors; check progress; zip the results; install a single missing package that a traceback names (and report it).
- **May not:** change any code (ours or the authors'), configs or training options; delete finished results; run anything other than round 4; "fix" a failure by changing settings.
