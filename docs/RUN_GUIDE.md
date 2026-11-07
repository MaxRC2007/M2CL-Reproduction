# What to run next, in order

Everything is ready. You run steps 1-6 yourself; Claude does step 7 and the paper updates.

## 1. Fix the plan publicly before running anything (about 10 minutes)
This step is what makes the test "pre-registered", which is the strongest thing in the paper. Skip it and the decisive test becomes just another experiment.

1. Put these files in a public GitHub repository (yours), commit and push:
   - `docs/PREREGISTRATION.md`
   - `configs/round3_m2.yaml`, `configs/round3_m2cl.yaml`, `configs/round3_erm.yaml`
   - `scripts/analyze_round3.py`
   - `notebooks/Round3_Decisive_Colab.ipynb`
   - `notebooks/Round3_Kaggle.ipynb`
   - the rest of the code (`src/`, `scripts/`, `tests/`), so the timestamped plan matches runnable code
2. Copy the commit hash into the paper at `\authorcheck{commit hash or OSF link}` in the section "Pre-Registered Decisive Test".
3. Optional but stronger: also upload `PREREGISTRATION.md` to https://osf.io as a registration. It gives a frozen, citable timestamp.

Don't change any setting after this point. If something must change, write down why and report it in the paper as a deviation from the pre-registration.

## 2. Upload to Google Drive (each account you'll use)
At the top level of My Drive:
- `M2CL_code.zip` from `M2CL-Reproduction\colab_upload\`. **It was rebuilt on 7 Nov 2026 and adds VLCS support.** The older copy is kept as `M2CL_code_2026-11-05.zip`; don't upload that one.
- `PACS.zip` (in `Downloads\Madhu\project\dataset\`)
- Optional, so the notebook can show the full analysis: `Downloads\lr.zip` and `Downloads\m2_v2.zip`

## 3. Run Part A first: it decides the paper (about 10 GPU hours)
Open `notebooks/Round3_Decisive_Colab.ipynb` in Colab (File > Upload notebook), choose Runtime > T4 GPU, and run the cells in order up to Part A.

- 8 M2 runs (lr 1e-3, seeds 1-2) and 12 M2-CL runs (alpha 0.07, lr 1e-3, seeds 0-2).
- Each run takes about 25-30 minutes; at about 3-4 hours of T4 time per free account, this spans 3-4 sessions.

**Switching accounts** (same as before): download the `m2cl_v2` folder from the old account's Drive, upload the zip plus the files from step 2 to the new account, and run all cells. Unfinished runs resume where they stopped.

Note which runs print a `[resume]` line.

## 4. Part B: ERM at lr 1e-3 (about 4-5 hours)
Needed for the paper's comparison with the baseline at the better learning rate.

## Part C runs on the institute GPU machine (Amendment 2)
Follow `docs/RUN_INSTITUTE.md`. The Kaggle section below is kept for the record. Kaggle ran only the first VLCS run, as a hardware check.

## Kaggle (superseded for Part C): part C, in parallel with Colab
Kaggle gives 30 GPU hours a week in sessions of up to 12 hours. The pre-registration (Amendment 1) assigns **C to Kaggle** and **A, B and D to Colab**, so every comparison stays on one platform and both platforms work at the same time. Part C needs about 15-18 GPU hours: 2 sessions.

**If you already pushed the pre-registration**, push the updated `docs/PREREGISTRATION.md` (Amendment 1) and `notebooks/Round3_Kaggle.ipynb` as a new commit before the first Kaggle run. The amendment is then timestamped before any of its results exist.

**One-time setup**
1. Kaggle > Settings > Phone verification. Without it you get no GPU and no internet.
2. Datasets > New Dataset > upload `M2CL_code.zip` (from `colab_upload\`) and `PACS.zip` (from `Downloads\Madhu\project\dataset\`). Name it `m2cl-inputs` and keep it private.
3. Notebooks > New Notebook > File > Import Notebook > `notebooks\Round3_Kaggle.ipynb`.
4. In the notebook's right panel:
   - Session options > Accelerator: **GPU T4 x2**. Not P100: the notebook stops on P100.
   - Internet: **on**.
   - Add Input > Your Work > Datasets > `m2cl-inputs`.

**Each session**
1. Click **Save Version**, then **Save & Run All (Commit)**, then Save. It runs in the background, so you can close the browser. It runs C and stops starting new runs at about 11 hours, so the output is always saved. Watch it under the notebook's Versions or Logs.
2. When the version finishes, open the notebook editor again. Add Input > Your Work > Notebooks > this notebook (its newest version), and remove the older one if it is there. Then Save Version again. The notebook copies the finished runs in and continues where it stopped.
3. Session 1 should end with about 22 of 36 C runs, and session 2 finishes C. The last cell shows the counts.

**At the end:** open the newest version > Output > download `m2cl_v2`, put it in `Downloads`, and tell Claude. Keep the Colab results separate; Claude merges them.

VLCS on Kaggle streams from the Hugging Face copy automatically (audit F15). There is nothing to upload for it.

## 5. Part C: VLCS replication (about 15 hours)
**On Kaggle (the plan): see the Kaggle section above; skip this step on Colab.** If you do run it on Colab, re-upload `notebooks/Round3_Decisive_Colab.ipynb` first: the VLCS cell was rewritten on 6 Nov 2026 after DomainBed's Google Drive link hit its download limit (`FileURLRetrievalError: Cannot retrieve the public link of the file`).

The VLCS cell tries three sources in order:
1. `VLCS.tar.gz` at the top of My Drive. That can be a file you uploaded, or a shortcut to DomainBed's file: open https://drive.google.com/file/d/1skwblH1_okBwxWxmRsp9_qi15hyPpxg8 signed in to the Colab account, then Organize > Add shortcut > My Drive, and name it `VLCS.tar.gz`.
2. DomainBed's link.
3. A Hugging Face mirror (`PTeterwak/DomainBed_OOP`, `all/VLCS.tar.gz`, 3.8 GB, saved only to the Colab disk).

Normally you do nothing and it falls through to the mirror. Whatever the source, the cell stops unless each domain has exactly DomainBed's image count (Caltech101 1,415, LabelMe 2,656, SUN09 3,282, VOC2007 3,376; 10,729 in total).

## 6. Part D: seeds 3-4 (optional, about 8 hours)
Runs on Colab. It is reported separately and never pooled into the primary test.

## 7. Send the results back
Download from Drive: `m2cl_v2/lr`, `m2cl_v2/round3`, `m2cl_v2/round3_vlcs` (or the whole `m2cl_v2` folder) and `m2cl_v2/round3_analysis.json`. Put them in `Downloads` and tell Claude, including any `[resume]` lines. Claude then:
- runs `scripts/analyze_round3.py` with `--grad` (the manipulation checks),
- fills the `[PENDING]` section of the paper with the pre-registered decision, whatever it is,
- updates the abstract, discussion and conclusion.

## 8. Before submitting
- [ ] Compile `paper/M2CL_v3.tex` on Overleaf with `fig_pairs.pdf` in the same folder; send Claude any errors.
- [ ] Resolve the two `\authorcheck` markers: the 61.06/61.03 provenance (search both Google Drives for `m2cl_checkpoints/` and `m2cl_history.json`; if they can't be found, Claude will mark them unverified) and the repository URL.
- [ ] Check these citations on IEEE Xplore, AAAI and CVF Open Access (not verifiable online by Claude): the volume, issue and pages for Ballas & Diou (IEEE TAI 2024), DomCLP (AAAI 2025) pages, SelfReg (ICCV 2021) pages, GradNorm, SupCon and Pineau et al.
- [ ] Email the authors (Aristotelis Ballas, Christos Diou) before submission: say you are reproducing M2-CL, ask for the exact command and hyperparameters behind Table I, and mention the gradient-share finding. Draft below; send it yourself.
- [ ] Show Dr. Prasath the draft and `docs/CLAIMS.md`.

### Draft email to the authors (edit before sending)
> Subject: Reproducing M2-CL (IEEE TAI 2024): question about the PACS ResNet-18 settings
>
> Dear Dr. Ballas and Dr. Diou,
>
> I am an undergraduate at IIIT Sri City, working under Dr. Rajendra Prasath on a reproduction of your M2-CL paper on PACS with ResNet-18. Your public code (commit 0b1bce3) was very helpful. Could you share the exact command or hyperparameters behind the ResNet-18 PACS results in Table I, in particular the batch size per domain, the number of steps, and the learning rate?
>
> One observation you may find interesting: in our reimplementation, which differs from yours (6 extraction blocks instead of 21 pipelines), the contrastive gradient was about 0.4 % of the cross-entropy gradient at our settings, against about 11-13 % in your code at its defaults. We are now testing at a matched strength.
>
> Thank you for your time.
>
> Best regards,
> Madhusudan

## Where things are
| What | File |
|---|---|
| Paper | `paper/M2CL_v3.tex`, `paper/fig_pairs.pdf` |
| Every claim and its source | `docs/CLAIMS.md` |
| Full investigation log | `docs/AUDIT.md` (sections A-F13) |
| Pre-registration | `docs/PREREGISTRATION.md` |
| Notebook for the next runs | `notebooks/Round3_Decisive_Colab.ipynb` |
| Analysis, fixed in advance | `scripts/analyze_round3.py` |
