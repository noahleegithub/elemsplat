# final_hprc: ECEN 689 final project on TAMU HPRC

Code and Slurm jobs for the final project *From UAV Images to Element-Level Bridge Geometry:
Layout-Guided Instance Segmentation with Gaussian Splatting*. Equation and section numbers refer
to the paper in `ECEN689_Final_Paper_latex.zip` (`main.tex`).

| Part | Question | Data | Jobs |
|---|---|---|---|
| Track A | RQ1, member identity | BrPCD point clouds (10 real scans + 88 variants) | 03a semantic nets, 03b M1/M2 |
| Track B | RQ2, geometry | VTTI UAV images + shipped COLMAP project FRO | 03c 3DGS, 03d 2DGS, 03g VGGT (feed-forward baseline), 03e measurement |
| Merge | RQ3, end to end | 2DGS points of VTTI | 03f |

Default cluster: **ACES** (H100 80 GB for 3DGS/2DGS, A30 24 GB for the point-cloud networks,
CPU partition for M1/M2 and measurement). Grace works with `CLUSTER=grace` (A100 40 GB everywhere).

## 1. Layout on $SCRATCH

```
$SCRATCH/ECEN689/
  elemsplat/         <- this package (git clone git@github.com:noahleegithub/elemsplat.git)
  envs/final_pc      <- point clouds, M1/M2, metrics, figures
  envs/final_gs      <- 3DGS + 2DGS CUDA extensions, pycolmap, open3d
  repos/             <- gaussian-splatting (commit 54c035f, as in HW1), 2d-gaussian-splatting, vggt
  final_data/        <- brpcd_raw, brpcd (converted), vtti_raw (zips), vtti, vtti_scene
  final_results/     <- everything the jobs write; paper_assets/ is what the paper needs
  final_smoke/       <- synthetic smoke test
```

## 2. Steps

| # | Where | Command | Time |
|---|---|---|---|
| 0 | login node | `bash 00_setup_env.sh` | 30 to 45 min (+ 5 GB of VGGT weights) |
| 1 | CPU job | `sbatch 00b_smoke_test.slurm` | 15 to 30 min |
| 2 | login node | `bash 01_download.sh` | 20 to 60 min (8.6 GB VTTI + BrPCD) |
| 3 | CPU job | `sbatch 02_prepare.slurm` | 0.5 to 2 h |
| 4 | Slurm chain | `bash run_all.sh` | 6 to 12 h wall clock |
| 5 | login node | `bash 04_eval_pack.sh` (also the last job of the chain) | minutes |

1. **Setup.** Run in a login-node shell (portal: Clusters, ACES Shell Access). A portal JupyterLab
   session is itself a Slurm job that charges SUs while it is open, so close it when done.
2. **Smoke test.** Runs every Python step on synthetic bridges and writes stamped figures to
   `final_smoke/paper_assets/figures`. The numbers are not results; the run only shows that the code works.
3. **Download.** VTTI files come from figshare with an md5 check (`VTTI_FILES` in `config.sh`);
   only the full-resolution images and the FRO project are needed. BrPCD comes from Google Drive with
   `gdown`. If Google refuses, download the three archives in a browser and upload them to
   `final_data/brpcd_raw/archives/`, then run `bash 01_download.sh brpcd` again.
4. **Prepare, then read the reports.**
   * `final_data/brpcd/inspect_report.txt`: every member file must map to a class (extend
     `configs/brpcd_classes.yaml` for names reported as UNMATCHED); the extents tell the unit
     (set `brpcd.unit_scale: 0.001` for millimetres); 10 real scans and 88 variants are expected.
   * `final_data/brpcd/index.csv`: column `source` gives the real scan of every variant. If some are
     marked `tie`, write `configs/brpcd_groups.csv` (columns `bridge,source`) and rerun 02 with `STEP=brpcd`.
   * `final_results/trackB/vtti_prepare.json`: registered images, camera model, mean reprojection
     error, number of test views, GPS tags, provisional scale.
5. **Run the chain.** `run_all.sh` submits 02 to 04 with dependencies (`SKIP_PREPARE=1` if 02 is done,
   `VGGT=0` to leave out the feed-forward baseline). Single jobs: `MODEL=pointnet2 sbatch --array=0-10 03a_semseg.slurm`,
   `sbatch 03c_gs3d.slurm`, `sbatch 03g_vggt.slurm`, ...
   VGGT runs once on at most `vggt.max_frames` images (one forward pass, about 10 min); its points are aligned to the
   COLMAP cameras and then measured like the others. `trackB/vggt/run_info.json` holds the camera-alignment residual
   (`residual_rms_m_provisional`): a few decimetres is normal, metres mean the frames disagree (check `rejected`).
6. **Check the measurement before trusting it.** Open `final_results/trackB/measure/diag_2dgs.png`:
   the blue boxes must contain the two piers and nothing else; dotted lines mark the ends. Correct
   boxes, end stations (abutment bearing lines if visible) and the girder count from the photos go into
   `configs/vtti_overrides.yaml`; then `NEW_FRAME=1 sbatch 03e_measure.slurm` and `sbatch 03f_merge.slurm`.
   `METHOD=vggt sbatch 03f_merge.slurm` repeats RQ3 on the VGGT points (optional; adds a column to the RQ3 table).
7. **Pack.** `04_eval_pack.sh` writes `final_results/paper_assets/{tables,figures,diagnostics}` and
   `final_results_pack.tar.gz`. Download it, copy `paper_assets/tables/*.tex` to the paper's `results/` and
   `paper_assets/figures/*` to its `figures/results/`, and recompile (`latexmk -pdf main`).
   Then write the blue `[...]` passages of the paper from the numbers.

## 2b. Datasets: where they come from and what to download

| Dataset | Used for | Source (download page) | Files the package needs | Licence |
|---|---|---|---|---|
| BrPCD (Zhang, Zhu, Xiong, Cai 2025, CACAIE 40(9), doi 10.1111/mice.13384) | Track A, RQ1 | https://github.com/wxiong12/SEU-Digital-Bridge -> Google Drive folder https://drive.google.com/drive/folders/1hoV1DeMOxDDwcQd7nwWBLmElzHuWlXk8 | all archives in the folder (10 real Leica P50 scans + 88 variants, one txt/ply per member, 6 classes) | article CC BY-NC-ND 4.0; dataset licence not stated (academic use, cite the paper) |
| VTTI Bridge (Sarlo and Ildefonzo 2025, doi 10.7294/29901590.v2) | Track B, RQ2 and RQ3 | https://data.lib.vt.edu/articles/dataset/Photogrammetry_Data_for_VTTI_Bridge_CIAMTIS_/29901590 (figshare file ids and md5 sums are in `VTTI_FILES` in `config.sh`) | `Full Resolution Reconstruction Images.zip` (2.7 GB), `FRO.zip` (5.6 GB, COLMAP project), plus the small `Image_Master_List.xlsx`, `COLMAP_Models_Options.xlsx`, `COLMAP_instructions.docx`, `README.rtf` | CC0 1.0 |
| SemanticBridge (Kellner et al. 2026) | fallback for RQ1 only if BrPCD cannot be used | https://github.com/mvg-inatech/3d_bridge_segmentation -> Zenodo records 18738225 and 18593422 | not downloaded by the package | CC BY-NC 4.0 |
| Boga Cayi UAV images (Gedik et al. 2026) | fallback for RQ2 only if VTTI fails to reconstruct | https://doi.org/10.5281/zenodo.22298893 | not downloaded by the package | CC BY 4.0 |

Software and weights fetched by `00_setup_env.sh`: https://github.com/graphdeco-inria/gaussian-splatting (commit
54c035f), https://github.com/hbb1/2d-gaussian-splatting, https://github.com/facebookresearch/vggt (commit a288dd0),
VGGT-1B weights from https://huggingface.co/facebook/VGGT-1B (research licence, non-commercial), LPIPS VGG weights
through torch hub.

How the download runs:

* `bash 01_download.sh` fetches BrPCD with `gdown --folder` into `final_data/brpcd_raw/archives/` and the needed
  VTTI files with `curl` from `https://ndownloader.figshare.com/files/<id>` into `final_data/vtti_raw/`, checks every
  md5 sum (three retries) and extracts the zips.
* Manual route for BrPCD (Google refuses `gdown` now and then): open the Drive folder in a browser, download every
  archive, upload them to `final_data/brpcd_raw/archives/` (portal Files app or `scp`), then `bash 01_download.sh brpcd`.
* Manual route for VTTI: download the two zips from the data page in a browser, keep the exact file names, upload them
  to `final_data/vtti_raw/`, then `bash 01_download.sh vtti` (it checks the md5 and skips the download).
* Not needed: the reduced-resolution VTTI images, the FRR/RRO/RRR projects and the UAV/phone query images (`needed=0`
  in `VTTI_FILES`; `VTTI_ALL=1 bash 01_download.sh` fetches everything, +23 GB). The fallbacks are not downloaded.

Which job reads what:

| Job | Data it reads | Comes from |
|---|---|---|
| `02_prepare` (STEP=brpcd) | BrPCD archives, one txt/ply per member | `final_data/brpcd_raw` |
| `02_prepare` (STEP=vtti) | `Full Resolution Reconstruction Images.zip` + `FRO.zip` | `final_data/vtti_raw` |
| `03a`, `03b` | `final_data/brpcd/*.npz` + `split.json` (written by 02) | BrPCD only |
| `03c`, `03d`, `03g` | `final_data/vtti_scene/` (undistorted 1,600 px images + `sparse/0`, written by 02) | VTTI only |
| `03e` | the point clouds of 03c/03d/03g | VTTI only |
| `03f` | `trackB/measure/aligned_2dgs.npz` + the Track A network and tuned parameters | both |

## 3. Analysis settings

All analysis settings are in `configs/experiment.yaml`; `make_tables.py` turns them into the
appendix table of the paper, so the paper always shows what was run. The main ones:

| Stage | Setting |
|---|---|
| BrPCD | voxel = 2 x median point spacing of the real scans (0.02 to 0.20 m); normals from 16 neighbours; horizontal PCA puts the axis on x |
| Splits | leave-one-bridge-out over the 10 real scans; variants follow their scan; 1 validation scan per fold (same type when possible) |
| Semantic nets | PointNet and PointNet++ (pure PyTorch); 10 m blocks, 4,096 points, batch 16, 20k iterations, AdamW 1e-3, class weights n_c^-0.5 |
| M1 | DBSCAN per class, eps in {0.1 ... 1.2} m and m_pts in {5, 10, 20, 40}, tuned per class on validation |
| M2 | deck RANSAC 0.08 m; KDE b_s in {0.5, 1, 2} m, b_t in {0.10, 0.15, 0.25} m, eta in {0.10, 0.15, 0.25}; cable RANSAC delta_c in {0.10, 0.15, 0.25} m |
| Metrics | PQ with IoU > 0.5; segments under 30 points = noise; 2,000 bootstrap resamples of whole bridges |
| VTTI | project FRO, undistorted, long side 1,600 px, every 8th image held out |
| 3DGS | 30k iterations, lambda 0.2, densification gradient 2e-4 until 15k |
| 2DGS | 30k iterations, lambda_d 100, lambda_n 0.05; TSDF voxel 5 cm, truncation 5 voxels, depth cut 60 m; median and expected depth |
| VGGT | VGGT-1B, one pass on at most 200 images at 518 px width, no camera input; depth unprojected with the predicted cameras, confidence above the 50th percentile, 3 M points; similarity fit to the COLMAP camera centres |
| Measurement | deck plane within 15 deg of the camera vertical; girder band -3.0 to -0.3 m; piers below -3.5 m; station = midpoint of the 5th and 95th percentiles; 2 m deck patches |

## 4. Outputs

| File | Paper item |
|---|---|
| `paper_assets/tables/res_semantic.tex` | semantic IoU per class (Track A) |
| `paper_assets/tables/res_instances.tex` | PQ/SQ/RQ, slope of PQ against log N, H1 interval |
| `paper_assets/tables/res_render.tex` | PSNR/SSIM/LPIPS, training time, Gaussians |
| `paper_assets/tables/res_measure.tex` | scale, spans, side-span errors, deck residual (3DGS, 2DGS, 2DGS mean, VGGT) |
| `paper_assets/tables/res_rq3.tex` | girder and pier counts, spans from M2 |
| `paper_assets/tables/tab_settings_auto.tex` | appendix settings table |
| `paper_assets/figures/fig_pq_vs_n.pdf` | PQ against N, M1 vs M2, per bridge type |
| `paper_assets/figures/fig_semantic_iou.pdf`, `fig_brpcd_examples.pdf` | Track A qualitative + IoU |
| `paper_assets/figures/fig_render.pdf`, `fig_measure.pdf` | 3DGS vs 2DGS |
| `paper_assets/figures/fig_vtti_instances.pdf` | VTTI members coloured by class and by M2 instance |
| `paper_assets/figures/fig_cross_section.pdf` | cross-section slice (t, h) at mid main span of every measured point cloud (3DGS, 2DGS, VGGT); the measured version of the expected-geometry slide |
| `rq3/vtti_m2_instances.ply` | coloured point cloud for CloudCompare (slides) |
| `trackB/vggt/run_info.json`, `cameras_aligned.npz` | VGGT frames, time, GPU memory, camera-alignment residuals |

Every figure is written as PDF (TrueType fonts, as NeurIPS requires) and PNG.

## 5. Notes and troubleshooting

* **Quota.** `$SCRATCH` has a file-count limit (about 250k). Two envs + repos use about 90k files.
  `90_free_space.sh zips|models|envs` frees space after packing (`models` also removes `trackB/vggt/predictions.npz`).
* **VGGT out of memory.** Lower `vggt.max_frames` in `experiment.yaml` (200 frames need about 40 GB in bf16).
  `vggt.enabled: false` removes the baseline from every table and figure.
* **`module purge` and conda.** Every script deactivates and reactivates the env after loading
  modules (the HW1 fix); do not activate an env by hand before `sbatch`.
* **Out of memory in 3DGS/2DGS.** Set `gs3d.data_device: cpu`, or `vtti.long_side_px: 1200` and rerun
  02 with `STEP=vtti`.
* **CUDA extensions fail to import on the GPU node.** 03c/03d rebuild them for the node's GPU.
  `TORCH_CUDA_ARCH_LIST` in `config.sh` is 8.0;9.0 on ACES (A30 + H100).
* **2DGS flag names.** 03d checks `train.py --help` for `--lambda_dist` (else `--lambda_distortion`).
* **No pier found.** Set `pier_boxes` in `configs/vtti_overrides.yaml` (metres in the frame written
  to `trackB/measure/frame.json`).
* **`$'\r': command not found`.** A script was saved with Windows line endings; run `dos2unix *.sh *.slurm`
  (or `sed -i 's/\r$//' <file>`).
* **`invalid partition` or `invalid generic resource`.** Partition and GPU names differ between clusters;
  check `sinfo -s` and `GPU_PARTITION`, `GPU_BIG`, `GPU_SMALL`, `CPU_PARTITION` in `config.sh`.
* **Not automated.** M3 (`m3.enabled`) and the two RQ3 fallbacks (fine-tuning on hand-labelled VTTI
  points, SAM masks) are not implemented; the girder reference count is read from the photos.
