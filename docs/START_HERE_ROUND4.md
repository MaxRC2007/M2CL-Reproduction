# START HERE: round 4 (the authors' own code) on the institute Linux machine

This brief is for the person running the machine **and for any AI coding assistant helping them** (Claude in Antigravity, or similar). Read all of it before running anything. On the machine this file sits in the Part C folder, `~/Desktop/S20240030357/M2CL`; the Part C brief (`START_HERE.md`, `AGENTS.md`, `CLAUDE.md`) is for Part C only.

## What this is
Rounds 1-3 tested the contrastive term of M2-CL (Ballas & Diou, IEEE TAI 2024) in our own reimplementation. **Round 4 runs the authors' released code itself** on PACS, with the contrastive weight `lparam` at 0.01 and at 0, paired by seed: 24 runs (2 settings x 4 test domains x 3 seeds). The plan was fixed and pushed to GitHub before any run (`docs/PREREGISTRATION_ROUND4.md`); read it.

The machine: NVIDIA RTX 4000 Ada (20 GB), Intel Core Ultra 9, Linux. **Start only after Part C (VLCS) has finished** (it printed `Done.`), so the two never share the GPU.

## Rules (they protect the pre-registration)
1. **Don't edit** anything in `configs/`, `src/`, `scripts/`, or the authors' code that the script downloads (`~/data/authors_m2cl_0b1bce3`). The script checks that the authors' code is the unmodified commit and refuses to run otherwise.
2. **Run only `scripts/run_round4_authors.py`** (and the tests). **Use only these options** of `run_round4_authors.py`: `--out`, `--data_dir`, `--parallel`, `--smoke`. Never pass `--no_gpu_check`.
3. **Don't delete or edit** anything in `~/m2cl_round4`. To continue after any stop, run the same command again. The authors' code cannot resume a run, so the script deletes an *unfinished* run and restarts it with the same seeds. Finished runs are always kept.
4. **If something fails, stop and report.** Collect the last 30 lines of the run's log and send them to the project owner. Don't "fix" it by changing code or settings. The only fixes allowed without the owner are installing the packages listed below.
5. Don't install or upgrade anything beyond the commands below. Don't change the GPU driver.

## Steps
This is an **add-on** to the Part C folder, now `~/Desktop/S20240030357/M2CL` (it was moved there from `M2CL_PartC_institute` during Part C; the old name is a symlink). It only adds new files. `unzip -n` never overwrites anything, so Part C's code, results and the F20 edit there stay exactly as they are.
```bash
# 0. Add the round-4 files to the Part C folder
cd ~/Desktop/S20240030357/M2CL && unzip -n ~/Downloads/M2CL_Round4_addon.zip
# 1a. The add-on arrived intact, and round 4's other helper (analyze_round3.py) is unchanged: every line must say OK
sha256sum -c ROUND4_SHA256SUMS.txt
# 1b. Nothing else from the Part C package was changed. Every file under src/, configs/, scripts/ or tests/ must say OK.
#     A FAILED .md file is only noted (send the output); a FAILED code file means stop and send the output.
sha256sum -c ROUND4_reference/PARTC_SHA256SUMS.txt
# 1c. The one known code edit (the RTX 4000 Ada GPU check, docs/AUDIT.md F20/F26), saved for the owner:
diff ROUND4_reference/run_part_c.py.orig scripts/run_part_c.py > ~/edits.diff; cat ~/edits.diff
#     It must show only the lines inside check_gpu() that accept sm_89 when sm_86 is listed. Round 4 reuses that
#     check, and the unedited one would refuse this GPU. If it shows anything else, stop and send ~/edits.diff.
# 2. Reuse the Python environment from Part C, then add the two packages the authors' code imports
source ~/m2cl-env/bin/activate   # the Part C environment (torch 2.11.0+cu128); use its new path if it was moved
pip install -r requirements_round4.txt
python -c "import torch, wilds, timm; print(torch.__version__, torch.cuda.get_device_name(0))"
# 3. Optional, quick: the tests (about 10 minutes; all 8 must pass)
python -m pytest -q tests/test_round4.py
# 4. In tmux (tmux new -s round4; detach Ctrl+B then D; return: tmux attach -t round4)
python scripts/run_round4_authors.py --out ~/m2cl_round4 --data_dir ~/data --smoke
#    downloads the authors' code (pinned commit) and DomainBed's PACS (0.18 GB, checks the 9,991 images),
#    trains 101 steps, then prints the time per step, the peak GPU memory and an estimate for all 24 runs.
# 5. All 24 runs. Use --parallel 2 if twice the smoke run's peak GPU memory fits well within 20 GB; else 1.
python scripts/run_round4_authors.py --out ~/m2cl_round4 --data_dir ~/data --parallel 2
# 6. When it prints "Done.":
cd ~ && cp ~/edits.diff m2cl_round4/ && zip -r m2cl_round4.zip m2cl_round4
```
**Don't move or rename folders while round 4 runs** (not the project folder, the environment, `~/data` or `~/m2cl_round4`). Moving the environment during Part C is what crashed two runs there (F26).

Upload `m2cl_round4.zip` to Google Drive and download it to the laptop's `Downloads` folder. It holds every run's `results.jsonl`, final model, logs, the hardware record (`env/`) and `edits.diff`.

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
