# START HERE: round 6 (the authors' own code on PACS, new seeds) on the institute Linux machine

This brief is for the person running the machine **and for any AI coding assistant helping them** (Claude in Antigravity, or similar). Read all of it before running anything. On the machine, this file sits in the project folder `~/Desktop/S20240030357/M2CL`, next to the files from rounds 4 and 5.

## What this is
Round 4 ran the authors' released M2-CL code on PACS with seeds 0-2. **Round 6 runs exactly the same code and settings on five new seeds, 3-7.** It tests one specific finding from round 4: at the final checkpoint, the contrastive term seemed to add about 0.7 pp.
- **Runs:** 40 runs. The contrastive weight `lparam` is 0.01 or 0, and the runs are paired by seed: 2 settings x 4 test domains x 5 seeds.
- **Plan:** fixed and pushed to GitHub before any run, in `docs/PREREGISTRATION_ROUND6.md`. Read it.
- **Reused from earlier rounds on this machine:** the authors' code from round 4 (`~/data/authors_m2cl_0b1bce3`), PACS from round 4 (under `~/data`), and the Python environment `~/m2cl-env`.
- **Machine:** the same as rounds 4 and 5, an NVIDIA RTX 4000 Ada (20 GB) on Linux.

## Rules (they protect the pre-registration)
1. **Don't edit** anything in `configs/`, `src/`, `scripts/` or the authors' code. The script checks that the authors' code is the unmodified commit, and refuses to run otherwise.
2. **Run only `scripts/run_round6_authors.py`** (and the tests), with only these options: `--out`, `--data_dir`, `--parallel`, `--smoke`. Never pass `--no_gpu_check`.
3. **Don't delete or edit** anything in `~/m2cl_round6`.
   - To continue after any stop, run the same command again.
   - An *unfinished* run is deleted and restarted with the same seeds. Finished runs are always kept.
4. **If something fails, stop and report.** Send the last 30 lines of the run's log to the project owner. Don't "fix" it by changing code or settings.
5. **Don't move or rename folders while it runs:** not the project folder, `~/m2cl-env`, `~/data` or `~/m2cl_round6`.
6. **Don't look at partial results and stop early.** The plan analyses all 40 runs together, once.

## Steps
```bash
# 0. Add the round-6 files (unzip -n never overwrites anything already there)
cd ~/Desktop/S20240030357/M2CL && unzip -n ~/Downloads/M2CL_Round6_addon.zip
# 1a. Check that the add-on arrived intact, and that the round-4 files round 6 reuses are unchanged.
#     Every line must say OK.
sha256sum -c ROUND6_SHA256SUMS.txt
# 1b. Check that the only code edit on this machine is still the known GPU check.
#     This must print "edit check OK".
diff ROUND4_reference/run_part_c.py.orig scripts/run_part_c.py | diff - ROUND6_reference/edits_expected.diff && echo "edit check OK"
# 2. Activate the environment used by rounds 4 and 5
source ~/m2cl-env/bin/activate
python -c "import torch, wilds, timm; print(torch.__version__, torch.cuda.get_device_name(0))"
# 3. Optional: run the tests (about 15-30 minutes; all 6 must pass)
python -m pytest -q tests/test_round6.py
# 4. Inside tmux (tmux new -s round6; detach with Ctrl+B then D; return with tmux attach -t round6), run the smoke test.
#    It finds the authors' code and PACS (checking the 9,991 images) and trains for 101 steps.
#    It then prints the time per step, the peak GPU memory and an estimate for all 40 runs.
#    Send that estimate to the owner.
python scripts/run_round6_authors.py --out ~/m2cl_round6 --data_dir ~/data --smoke
# 5. Run all 40, two at a time as in round 4
python scripts/run_round6_authors.py --out ~/m2cl_round6 --data_dir ~/data --parallel 2
# 6. When it prints "Done.", zip the results
cd ~ && zip -r m2cl_round6.zip m2cl_round6
```

**Expected time:** round 4 took about 17.5 hours for 24 PACS runs with `--parallel 2`, so 40 runs should take about **29-30 hours**. The smoke test's estimate is the better guide.

Upload `m2cl_round6.zip` to Google Drive and download it to the laptop's `Downloads` folder. It contains:
- every run's `results.jsonl` and final model;
- the logs;
- the hardware record (`env/`).

## How to check progress
- **The tmux window** shows `[start]` lines and `[done] <run>: <accuracy>% ...` lines. That accuracy is the validation-selected one, shown only for information. The plan's primary result is computed at the end.
- **Finished runs:** run `ls ~/m2cl_round6/round6/*/`. A run is finished when its folder contains a file called `done`.
- **One run's log:** `tail -f ~/m2cl_round6/logs/<run>.log`. To see GPU use, run `nvidia-smi`.

## Known problems and what to do
| Symptom | Cause | What to do |
|---|---|---|
| `wrong image counts` or `has folders [...]` for PACS | `~/data` holds a different PACS copy | Stop and report the line; don't move or delete data |
| `CUDA out of memory` | Two runs don't fit | Press Ctrl+C, then restart with `--parallel 1`; finished runs are kept |
| `Bus error` / `DataLoader worker ... killed` | Too little shared memory | Ask the admin to raise shared memory; don't change the code |
| `not the pre-registered unmodified commit` | The authors' folder was changed | Stop and report; don't delete it without the owner |
| Any other traceback | Unknown | Stop. Send the last 30 lines of the run's log to the owner; don't edit code |

## What the AI assistant may and may not do here
- **May:** run the commands above; read logs and explain errors; check progress; zip the results.
- **May not:**
  - change any code (ours or the authors'), configs or training options;
  - delete finished results or data;
  - move folders;
  - run anything other than round 6;
  - analyse or report partial results.
