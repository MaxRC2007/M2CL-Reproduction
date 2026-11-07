# Part C (VLCS) on the institute GPU machine (Linux)

Part C is 36 runs: M2, M2-CL and ERM × 4 VLCS folds × seeds 0-2. Pre-registration Amendment 2 moves all of them to the institute machine. Parts A, B and D stay on Colab.

## 0. Before the first run (once)
1. **Kaggle:** stop the session (Cancel run) and run nothing else on Kaggle. If its log already shows `[done] m2_6tap_Caltech101_seed0: target Caltech101 = ...%`, copy that line and send it to Claude; it is the hardware check. The Kaggle run is not continued here: this machine runs all 36 runs from the start.
2. **Push the pre-registration, Amendment 2 and `scripts/run_part_c.py`** to GitHub as a new commit before step 4 below. The timestamp shows the move was decided before any institute result existed.

## 1. Copy the code to the machine
Upload `Downloads\M2CL_PartC_institute.zip` from the laptop to Google Drive, and download it on the machine. It holds the code, configs, tests and docs, including `START_HERE.md` for an AI assistant.

```bash
mkdir -p ~/m2cl && cd ~/m2cl
unzip ~/Downloads/M2CL_PartC_institute.zip   # wherever the download went
ls scripts/run_part_c.py START_HERE.md        # both should exist
```

## 2. Python environment (once)
Check the driver first: `nvidia-smi` must show the GPU and a **CUDA version of 12.8 or higher** in its top-right corner. Blackwell GPUs need it; if it is lower, ask the lab admin to update the driver.

With conda:
```bash
conda create -y -n m2cl python=3.11
conda activate m2cl
```
Without conda:
```bash
python3 -m venv ~/m2cl-env && source ~/m2cl-env/bin/activate
```
Then, in either case:
```bash
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r ~/m2cl/requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0), torch.cuda.get_arch_list())"
```
The last line must print a version of 2.7 or newer, the GPU name, and a list that includes `sm_120`.

## 3. Smoke test (about 5-10 minutes plus the 3.8 GB VLCS download)
Use tmux so the work continues when you log out or the connection drops:
```bash
tmux new -s partc
conda activate m2cl            # or: source ~/m2cl-env/bin/activate
cd ~/m2cl
python scripts/run_part_c.py --out ~/m2cl_v2 --data_dir ~/data --smoke
```
It checks the GPU, downloads VLCS once into `~/data`, checks DomainBed's image counts (10,729), and trains one epoch of M2-CL in a separate test folder (`~/m2cl_v2_smoke`, never analysed). It ends with `Smoke test passed` and an estimate of the total time. If anything fails, send Claude the last 20 lines.

While it runs, open a second tmux window (Ctrl+B then C) and run `nvidia-smi`. Note how much GPU memory the run uses.

## 4. All 36 runs
```bash
python scripts/run_part_c.py --out ~/m2cl_v2 --data_dir ~/data --parallel 2
```
- `--parallel 2` trains two runs at once on the one GPU. Use it if the smoke run used less than half the GPU memory; otherwise use `--parallel 1`. The runs are independent processes, so this does not change any result. It only makes each run a little slower and the whole part faster.
- **Detach** and leave it running: Ctrl+B then D. **Come back** later: `tmux attach -t partc`.
- **Progress:** each run prints `[start]` and `[done] <run>: <accuracy>% in <minutes> min`. Full logs are in `~/m2cl_v2/logs/`; follow one with `tail -f ~/m2cl_v2/logs/<run>.log`.
- **If it stops** (failure, reboot, Ctrl+C), run the same command again. Finished runs are skipped and unfinished ones resume from their last epoch. Don't delete anything in `~/m2cl_v2`.
- Don't start a second copy for the same folder; the script refuses to.

When all 36 are done it runs the pre-registered analysis and prints `Done.`

## 5. Send the results back
```bash
cd ~ && zip -r m2cl_v2_C.zip m2cl_v2
```
Copy `m2cl_v2_C.zip` (about 2 GB, mostly the saved models) to the laptop's `Downloads` folder and tell Claude. It holds every run, the logs, the hardware record (`env/`) and the list of truncated VLCS images.

## Don't
- Don't change any setting in `configs/round3_*.yaml` or any command-line option other than `--parallel`.
- Don't run any Part C run anywhere else.
- Don't use the smoke folder for anything.
