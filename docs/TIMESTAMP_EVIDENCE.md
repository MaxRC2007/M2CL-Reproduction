# Timing evidence for the round-3 plan

This file and its commit record what existed and when. Read it with `docs/PREREGISTRATION.md` and `docs/AUDIT.md` (F13-F22).

## Honest summary
- The round-3 plan was **not** deposited in a public registry before the PACS runs (Parts A, B, D).
- It was written, and its analysis code frozen, **before** those runs started. The evidence is below; part of it is held by Google, not by the author.
- This repository was first pushed publicly at 00:31 IST on 8 Nov 2026. **Correction (10 Nov 2026):** an earlier version of this file said Part C (VLCS) was still running at that time. The Part C files show it had finished at 23:15 IST on 7 Nov, and the institute script wrote the VLCS comparison at 23:15:30, 1 h 16 min before the push.
- The public push therefore does not protect Part C. What does is that the VLCS analysis used `scripts/analyze_round3.py` with the SHA-256 below. That file is byte-identical to the copy uploaded to Google Drive on 6 Nov 2026, before any round-3 run.
- The institute run used the shipped `run_part_c.py` (same hash). Whether the owner saw the printed VLCS result before the push is for the owner to state (`docs/AUDIT.md` F26).

## When each thing happened (IST, UTC+5:30)
| When | What | Evidence |
|---|---|---|
| 6 Nov 2026, 21:25-21:28 | `configs/round3_*.yaml` and `scripts/analyze_round3.py` written. The script sets the decision rules: margin 0.71 pp, inert distance 0.033, 10-of-12 rule, gradient share 0.05, sign-flip and TOST at 0.05 | File times; byte-identical copies inside `colab_upload/M2CL_code_2026-11-06.zip` and every later code zip |
| 6 Nov 2026, 21:26 | `docs/PREREGISTRATION.md` sections 1-9 written | Assistant tool log. The text as written then is `docs/PREREGISTRATION_as_of_2026-11-06.md`; sections 1-9 are unchanged since |
| 6 Nov 2026, evening | `M2CL_code.zip` uploaded to Google Drive on each Colab account | Google Drive upload dates, held by Google (screenshots to be added under `docs/evidence/`) |
| 6 Nov 2026, about 23:11 | First round-3 output (M2, lr 1e-3, Art, seed 1) | Drive download zip times, which appear to be US Pacific (UTC-8) |
| 7 Nov 2026 | Amendment 1 (platform split) and Amendment 2 (Part C moves to the institute machine) | Text in `docs/PREREGISTRATION.md` |
| 7 Nov 2026, 17:20-23:15 | Part C (VLCS, 36 runs) on the institute machine; the institute analysis is written at 23:15:30 | `env/` file names (UTC) and zip entry times in `checkpoints_archive/08_2026-11_round3_vlcs_C.zip` |
| 8 Nov 2026, 00:31 (19:01 UTC on 7 Nov) | Repository created and first pushed to https://github.com/MaxRC2007/M2CL-Reproduction (commit 65c8a5b, with README update 7971ae5). Part C had already finished (corrected 10 Nov) | GitHub `pushed_at` 2026-11-07T19:01:56Z (GitHub API) |
| 8 Nov 2026, 01:05 (19:35 UTC on 7 Nov) | Round-4 pre-registration pushed (commit 715a2d1), before any round-4 run | GitHub commit and push |
| 10 Nov 2026, 16:25 (10:55:37 UTC) | Torch-version check plan and its analysis script `scripts/analyze_torchcheck.py` pushed (commit d825855), with the verdict rule fixed (audit F28) | GitHub commit and push |
| 10 Nov 2026, 18:40 (13:10 UTC) | Torch-check reruns start on Colab, after the plan was public | `env_20261110T131006Z.json` (2.14.1) and `env_20261110T131008Z.json` (2.11.0) in the downloaded result zips |

## SHA-256 of the frozen files
This repository stores files byte for byte (`.gitattributes`: `* -text`), so these hashes can be checked on a checkout.

```
ba39e011851c9505dbe6db91d87eebb912b28be6623011d64b58daabfab24851  scripts/analyze_round3.py
d87023758e54689b1010132411f7e18ba3aa6be7cea78a7d158ebd8a4ba8e1b8  configs/round3_m2.yaml
cc07e5493d70d41751a44d49d06e97b766487ff622ca39545696bd1ca2552661  configs/round3_m2cl.yaml
3d88d8937ada7801b749a7720d27c5eb15497588335e18fbe97390faaa8cf577  configs/round3_erm.yaml
7c285b2d84a903e031dab03878281f0e005dd19ab6530a2e8fe3158e6f298a04  docs/PREREGISTRATION_as_of_2026-11-06.md
75a9ed715511aa102c9b3bc62bd5f66f24e05a6d152f128a470d0e2a9a4bb42b  scripts/run_part_c.py
54f9df66cac77c83bcab29d920a054a3f1411671bd50b0c44de379a7cdefb666  colab_upload/M2CL_code_2026-11-06.zip
a20dc9fce81633a91b00c41aeee46c252e630f928d5e8a944600a43df851fd37  colab_upload/M2CL_code_2026-11-07.zip
418703f38a0f8c09992da12d19e0bb27f0fbcadc78e5c4c1438a5b58c8e3dd55  colab_upload/M2CL_code.zip
```

## Known deviations recorded since (see `docs/AUDIT.md`)
- F20/F26: the institute GPU is an RTX Ada card, not Blackwell. The only code edit there was to the GPU check in `scripts/run_part_c.py` (to accept `sm_89`); it is not in the training, data or analysis path.
- F26: two VLCS runs crashed at epoch 22 after the project folder and Python environment were moved mid-run, and they resumed. Both selected epoch 19, and the data were not changed.
- F21: Colab moved from torch 2.11.0 to 2.14.1 during Part A.
