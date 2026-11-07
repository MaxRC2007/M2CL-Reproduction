# START HERE: Part C (VLCS) on the institute Linux machine

This brief is for the person running the machine **and for any AI coding assistant helping them** (Claude in Antigravity, Claude Code, or similar). Read all of it before running anything.

## What this project is
An honours research project (IIIT Sri City, advisor Dr. Rajendra Prasath): a pre-registered reproduction of M2-CL (Ballas & Diou, IEEE TAI 2024), a multiscale, multilayer contrastive-learning method for domain generalization. Round 3 is pre-registered in `docs/PREREGISTRATION.md`; read it, including Amendments 1 and 2. Parts A, B and D run on Google Colab. **This machine runs only Part C:** the VLCS replication, 36 training runs (M2, M2-CL and ERM × 4 VLCS target domains × seeds 0-2), with `scripts/run_part_c.py`.

## Rules (they protect the pre-registration; breaking one invalidates results)
1. **Don't edit** anything in `configs/`, `src/` or `scripts/`. If something crashes, stop, collect the error and report it to the project owner. A fix must be written up as a documented deviation before any run uses it.
2. **Use only these options** of `run_part_c.py`: `--out`, `--data_dir`, `--parallel`, `--smoke`. Never pass training settings such as epochs, learning rate, batch size or number of workers. Never use `--no_gpu_check`; it is for testing the script only.
3. Run **only Part C** here, and run each Part C run only here. Never train PACS runs (Parts A, B, D) on this machine.
4. **Don't delete or edit** anything inside the results folder (`~/m2cl_v2`). To continue after any stop, run the same command again: finished runs are skipped and unfinished runs resume from their last saved epoch.
5. **Start fresh.** A Kaggle session ran the first VLCS run for about an hour before this machine was available. That run is not continued here; this machine runs all 36 runs from the start. If the Kaggle run finished, its accuracy serves only as a hardware check (Amendment 2).
6. Don't install or upgrade packages beyond the commands below, and don't change the GPU driver yourself.

## Steps
Full detail with explanations: `docs/RUN_INSTITUTE.md`. Short version:

```bash
# 0. Unzip the package (downloaded from Google Drive) into ~/m2cl
mkdir -p ~/m2cl && cd ~/m2cl && unzip ~/Downloads/M2CL_PartC_institute.zip
# 1. Driver: the top right of nvidia-smi must say CUDA Version 12.8 or higher (needed for Blackwell)
nvidia-smi
# 2. Environment (once). With conda; without conda use: python3 -m venv ~/m2cl-env && source ~/m2cl-env/bin/activate
conda create -y -n m2cl python=3.11 && conda activate m2cl
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0), torch.cuda.get_arch_list())"
#    must print torch 2.7 or newer, the GPU name, and a list containing sm_120
# 3. Optional but quick: the unit tests (about 5 minutes; all must pass)
python -m pytest -q tests
# 4. In tmux, so the work survives logging out: tmux new -s partc   (detach: Ctrl+B then D; return: tmux attach -t partc)
python scripts/run_part_c.py --out ~/m2cl_v2 --data_dir ~/data --smoke
#    downloads VLCS once (3.8 GB), checks the 10,729 images, trains 1 epoch in ~/m2cl_v2_smoke (a test only)
#    and ends with "Smoke test passed" and a time estimate. While it trains, check GPU memory in nvidia-smi.
# 5. All 36 runs. Use --parallel 2 if the smoke run used under half the GPU memory, else --parallel 1.
python scripts/run_part_c.py --out ~/m2cl_v2 --data_dir ~/data --parallel 2
# 6. When it prints "Done.":
cd ~ && zip -r m2cl_v2_C.zip m2cl_v2
```
Upload `m2cl_v2_C.zip` to Google Drive and download it to the laptop's `Downloads` folder.

## How to check progress
- The tmux window shows `[start]` and `[done] <run>: <accuracy>% in <minutes> min` lines.
- `ls ~/m2cl_v2/round3_vlcs/*/` lists the run folders. A run is finished when its folder has `result.json`.
- `tail -f ~/m2cl_v2/logs/<run>.log` shows the epochs of one run.
- `nvidia-smi` shows GPU use.

## Known problems and what to do
| Symptom | Cause | What to do |
|---|---|---|
| `needs sm_120, which this PyTorch ... was not built for` or "no kernel image" | PyTorch too old for Blackwell | Reinstall with the cu128 command above |
| `nvidia-smi` shows CUDA below 12.8 | Driver too old | Ask the lab admin to update the NVIDIA driver; don't do it yourself |
| `CUDA out of memory` | Two runs don't fit | Ctrl+C, then restart with `--parallel 1`; nothing is lost |
| `Bus error` or `DataLoader worker ... killed` / shared memory errors | Too little `/dev/shm` (common in containers) | Ask the admin to raise shared memory; don't change the number of workers |
| `wrong image counts` | Bad or partial VLCS download | Delete `~/data/VLCS_raw.partial` (only that) and run again |
| `Another run_part_c.py ... is already using` | A second copy was started | Use the first one (`tmux attach -t partc`) |
| Any other traceback | Unknown | Stop. Save the last 30 lines of the run's log and send them to the project owner; don't edit code |

## What the AI assistant may and may not do here
- **May:** run the commands above; read logs and explain errors; check progress; zip the results; read any file in this package.
- **May not:** change code, configs or training options; delete results; run anything other than Part C; "fix" a failure by changing settings. Report failures instead; the project owner decides any change and records it in `docs/AUDIT.md`.
