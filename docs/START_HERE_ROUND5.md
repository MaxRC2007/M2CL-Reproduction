# START HERE: round 5 (the authors' own code on VLCS) on the institute Linux machine

This brief is for the person running the machine **and for any AI coding assistant helping them** (Claude in Antigravity, or similar). Read all of it before running anything. On the machine this file sits in the project folder, `~/Desktop/S20240030357/M2CL`, next to round 4's files.

## What this is
Round 4 ran the authors' released M2-CL code on PACS. **Round 5 runs the same code, with the same settings, on VLCS.**
- **Runs:** 24 runs, with the contrastive weight `lparam` at 0.01 and at 0, paired by seed (2 settings x 4 test domains x 3 seeds).
- **Plan:** fixed and pushed to GitHub before any run (`docs/PREREGISTRATION_ROUND5.md`); read it.
- **Inputs:** it reuses what is already on the machine: the authors' code from round 4 (`~/data/authors_m2cl_0b1bce3`), VLCS from Part C (`~/data/VLCS_raw`) and the Python environment `~/m2cl-env`.
- **Machine:** the same as rounds 3 and 4, an NVIDIA RTX 4000 Ada (20 GB) with an Intel Core Ultra 9, on Linux.

## Rules (they protect the pre-registration)
1. **Don't edit** anything in `configs/`, `src/`, `scripts/` or the authors' code. The script checks that the authors' code is the unmodified commit and refuses to run otherwise.
2. **Run only `scripts/run_round5_authors.py`** (and the tests), with only these options: `--out`, `--data_dir`, `--parallel`, `--smoke`. Never pass `--no_gpu_check`.
3. **Don't delete or edit** anything in `~/m2cl_round5`.
   - To continue after any stop, run the same command again.
   - An *unfinished* run is deleted and restarted with the same seeds. Finished runs are always kept.
4. **If something fails, stop and report.** Send the last 30 lines of the run's log to the project owner. Don't "fix" it by changing code or settings.
5. **Don't move or rename folders while it runs:** not the project folder, `~/m2cl-env`, `~/data` or `~/m2cl_round5`. Moving the environment is what crashed two Part C runs.

## Steps
```bash
# 0. Add the round-5 files (unzip -n never overwrites anything already there)
cd ~/Desktop/S20240030357/M2CL && unzip -n ~/Downloads/M2CL_Round5_addon.zip
# 1a. The add-on arrived intact, and round 4's files that round 5 reuses are unchanged: every line must say OK
sha256sum -c ROUND5_SHA256SUMS.txt
# 1b. The only code edit on this machine is still the known GPU check; this must print nothing
diff ROUND4_reference/run_part_c.py.orig scripts/run_part_c.py | diff - ROUND5_reference/edits_expected.diff && echo "edit check OK"
# 2. The Part C / round 4 environment
source ~/m2cl-env/bin/activate
python -c "import torch, wilds, timm; print(torch.__version__, torch.cuda.get_device_name(0))"
# 3. Optional: the tests (about 10-20 minutes; all 7 must pass)
python -m pytest -q tests/test_round5.py
# 4. In tmux (tmux new -s round5; detach Ctrl+B then D; return: tmux attach -t round5)
python scripts/run_round5_authors.py --out ~/m2cl_round5 --data_dir ~/data --smoke
#    finds the authors' code and VLCS (checks the 10,729 images), trains 101 steps, then prints the time per step,
#    the peak GPU memory and an estimate for all 24 runs. Note the estimate and send it to the owner.
# 5. All 24 runs. Use --parallel 2 if twice the smoke run's peak GPU memory fits well within 20 GB; else 1.
python scripts/run_round5_authors.py --out ~/m2cl_round5 --data_dir ~/data --parallel 2
# 6. When it prints "Done.":
cd ~ && zip -r m2cl_round5.zip m2cl_round5
```

**Expected time:** round 4 (PACS) took about 17.5 hours with `--parallel 2`. VLCS images are larger and slower to decode, so expect about a day. The smoke test's estimate is the better guide.

Upload `m2cl_round5.zip` to Google Drive and download it to the laptop's `Downloads` folder. It holds:
- every run's `results.jsonl` and final model;
- the logs;
- the hardware record (`env/`), with the list of truncated VLCS images.

## How to check progress
- The tmux window shows `[start]` and `[done] <run>: <accuracy>% ...` lines.
- `ls ~/m2cl_round5/round5/*/`: a run is finished when its folder has a file called `done`.
- `tail -f ~/m2cl_round5/logs/<run>.log` follows one run; `nvidia-smi` shows GPU use.

## Known problems and what to do
| Symptom | Cause | What to do |
|---|---|---|
| `wrong image counts` or `has folders [...]` for VLCS | `~/data` holds a different VLCS copy, or extra folders inside `VLCS/` | Stop and report the line; don't move or delete data |
| `CUDA out of memory` | Two runs don't fit | Ctrl+C, then restart with `--parallel 1`; finished runs are kept |
| `Bus error` / `DataLoader worker ... killed` | Too little shared memory; the authors' code uses 8 loader workers | Ask the admin to raise shared memory; don't change the code |
| `not the pre-registered unmodified commit` | The authors' folder was changed | Stop and report; don't delete it without the owner |
| Very slow progress, GPU use low | VLCS JPEG decoding on the CPU | Expected; let it run |
| Any other traceback | Unknown | Stop. Send the last 30 lines of the run's log to the owner; don't edit code |

## What the AI assistant may and may not do here
- **May:** run the commands above; read logs and explain errors; check progress; zip the results.
- **May not:** change any code (ours or the authors'), configs or training options; delete finished results or data; move folders; run anything other than round 5.
