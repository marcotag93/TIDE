<p align="center">
  <img src="https://raw.githubusercontent.com/marcotag93/TIDE/v1.30.0/src/tide/assets/logo.png" alt="TIDE logo" width="280"/>
</p>

<h1 align="center">TIDE</h1>

<p align="center">
  <b>Tractography-Informed Dose Estimation</b><br>
  Individualised TMS intensity estimation from subject-specific tractography and SimNIBS electric-field modelling
</p>

<p align="center">
  <code>SimNIBS</code> • <code>diffusion MRI tractography</code> • <code>activating function</code> • <code>TMS dosing</code>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11--3.12-3776AB.svg" alt="Python 3.11-3.12"/>
  <a href="https://pypi.org/project/tide-pipeline/"><img src="https://img.shields.io/pypi/v/tide-pipeline.svg" alt="PyPI version"/></a>
  <img src="https://img.shields.io/badge/SimNIBS-reference%204.5-2E8B57.svg" alt="SimNIBS reference environment 4.5"/>
  <img src="https://img.shields.io/badge/status-beta-orange.svg" alt="Beta status"/>
  <a href="https://github.com/marcotag93/TIDE/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-GPL--3.0-blue.svg" alt="GPL-3.0 license"/></a>
  <a href="#research-use-only"><img src="https://img.shields.io/badge/use-research%20only-B31B1B.svg" alt="Research use only"/></a>
</p>

> [!IMPORTANT]
> <a id="research-use-only"></a>
> **Research use only.** TIDE is research software for computational TMS modelling. It is **not a medical device**, has not been clinically validated for individual treatment decisions, and has no regulatory clearance. It must not be used for diagnosis, treatment planning, or clinical decision-making. The activating function and tractography-derived quantities used by TIDE are model-based proxies, not direct measurements of axonal recruitment.

---

## 👤 Author

**Marco Tagliaferri** — *PhD Candidate in Neuroscience*
🏛️ [Center for Mind/Brain Sciences (CIMeC)](https://www.cimec.unitn.it/), University of Trento, Italy

[![Email](https://img.shields.io/badge/Email-marco.tagliaferri%40unitn.it-D14836?style=flat&logo=gmail&logoColor=white)](mailto:marco.tagliaferri@unitn.it)
[![Email](https://img.shields.io/badge/Email-marco.tagliaferri93%40gmail.com-D14836?style=flat&logo=gmail&logoColor=white)](mailto:marco.tagliaferri93@gmail.com)
[![ORCID](https://img.shields.io/badge/ORCID-0000--0002--1800--3977-A6CE39?style=flat&logo=orcid&logoColor=white)](https://orcid.org/0000-0002-1800-3977)
[![GitHub](https://img.shields.io/badge/GitHub-marcotag93-181717?style=flat&logo=github)](https://github.com/marcotag93)

If you use TIDE in your research, please cite the accompanying preprint:

**APA:**

> Tagliaferri, M., Cattaneo, L., Miniussi, C., & Brancaccio, A. (2026). *TIDE: Tractography-Informed Dose Estimation for individualised TMS intensity*. **bioRxiv**. DOI: `10.1101/<BIOARXIV_DOI>`

**BibTeX:**

```bibtex
@article{Tagliaferri_TIDE_2026,
  author  = {Tagliaferri, Marco and Cattaneo, Luigi and Miniussi, Carlo and Brancaccio, Arianna},
  title   = {{TIDE}: Tractography-Informed Dose Estimation for individualised TMS intensity},
  journal = {bioRxiv},
  year    = {2026},
  doi     = {10.1101/<BIOARXIV_DOI>},
  url     = {https://doi.org/10.1101/<BIOARXIV_DOI>},
  note    = {Preprint}
}
```

> [!NOTE]
> **Manuscript status.** The manuscript describing TIDE is currently in preparation. The peer-reviewed article citation will replace the preprint citation once the manuscript is formally published.

The TIDE software release associated with this work is archived on Zenodo under DOI [`10.5281/zenodo.22019737`](https://doi.org/10.5281/zenodo.22019737). Software metadata and the preferred preprint citation are provided in [`CITATION.cff`](https://github.com/marcotag93/TIDE/blob/main/CITATION.cff) and are available through GitHub's **Cite this repository** function.

### SimNIBS citations

TIDE uses **SimNIBS** as its finite-element electric-field modelling and TMS simulation backend. Therefore, publications using TIDE should **cite the TIDE preprint above as the primary method citation** and additionally cite the relevant SimNIBS publication(s) for the simulation components used.

For the SimNIBS TMS modelling framework, please cite:

> Thielscher, A., Antunes, A., & Saturnino, G. B. (2015). Field modeling for transcranial magnetic stimulation: A useful tool to understand the physiological effects of TMS? *37th Annual International Conference of the IEEE Engineering in Medicine and Biology Society (EMBC)*, 222–225. https://doi.org/10.1109/EMBC.2015.7318340

If your TIDE configuration uses **Auxiliary Dipole Method (ADM) coil-position optimisation** (`options.adm_optimization: true`), please also cite:

> Gomez, L. J., Dannhauer, M., & Peterchev, A. V. (2021). Fast computational optimization of TMS coil placement for individualized electric field targeting. *NeuroImage, 228*, 117696. https://doi.org/10.1016/j.neuroimage.2020.117696

For analyses relying on other SimNIBS-specific modules, head-model pipelines, or coil datasets, please follow the corresponding module-specific citation guidance in the SimNIBS documentation.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [How TIDE Works](#how-tide-works)
- [Getting Started](#getting-started)
- [Configuration](#configuration)
- [Workflows](#workflows)
- [Python API](#python-api)
- [Outputs](#outputs)
- [Advanced Usage](#advanced-usage)
- [For Developers](#for-developers)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [Contact](#contact)

---

## Overview

**TIDE** is an open-source, SimNIBS-based pipeline for estimating an individualised TMS intensity for a tractography-defined target pathway. It combines:

1. an empirical **resting motor threshold (RMT)** measured at motor cortex;
2. subject-specific **finite-element electric-field modelling**;
3. diffusion MRI **tractography** of the corticospinal tract (CST) and target pathway;
4. the gradient-term **activating function (AF)** evaluated along streamlines.

The CST at the motor hotspot provides the calibration reference. TIDE estimates the target intensity required to reproduce the bundle-level AF efficiency observed in the CST at the measured RMT.

TIDE is designed for **research on pathway-informed TMS dosing**. It does not model the full nonlinear biophysics of axonal excitation, and tractography streamlines must not be interpreted as direct anatomical measurements of individual axons.

---

## ✨ Key Features

### 🎯 Individualised intensity estimation

Estimate a target-specific stimulation intensity in **% of maximum stimulator output** using each participant's measured RMT as the empirical calibration anchor.

### 🧠 Tractography-informed electric-field analysis

Sample the SimNIBS vector E-field along subject-specific streamlines and compute the gradient activating function:

```text
AF = d(E · T) / ds
```

where `E` is the electric-field vector, `T` is the local streamline tangent, and `s` is physical arc length.

### 📏 Arc-length AF implementation

Geometry and E-field samples are jointly interpolated to a common physical support (≤ 0.5 mm spacing), smoothed using a 2.5 mm physical Gaussian scale, and differentiated along arc length. AF polarity is preserved internally; magnitude is used for activation-threshold aggregation.

### 🗺️ Grid search

Evaluate multiple candidate target positions and generate a spatial map of TIDE-estimated intensity, together with per-point reproducibility configurations and QC information.

### ⚖️ Weighted and surface-constrained analyses

Optionally incorporate **SIFT2 streamline weights** and a FreeSurfer grey-white interface surface. Weighted and unweighted results are both retained for auditability.

### 🧭 Neuronavigation export

After a successful estimation, optionally append the final target pose to a **Softaxic `.stmpx`** template.

### 📊 Reproducible reports and visualisation

Generate human-readable TXT reports, structured JSON sidecars, self-contained HTML reports, tractogram/NIfTI derivatives, optional 3D visualisations, and replayable YAML configurations.

### ⚡ Exact fixed-pose cache

Reuse deterministic SimNIBS results for identical fixed coil poses through a content-addressed cache without caching or approximating downstream AF calculations.

---

## How TIDE Works

At a high level, TIDE applies the same numerical core to the **motor calibration pathway** and the **target pathway**:

```text
CST at M1                                  Target pathway
   │                                             │
   ├─ coil pose / optimisation                   ├─ coil pose / optimisation
   ├─ SimNIBS FEM E-field                        ├─ SimNIBS FEM E-field
   ├─ vector E-field sampling                    ├─ vector E-field sampling
   ├─ AF = d(E·T)/ds                             ├─ AF = d(E·T)/ds
   └─ bundle-level AF metric                     └─ bundle-level AF metric
                 │                                  │
                 └──────────────┬───────────────────┘
                                │
                         RMT calibration
                                │
                                ▼
                  target intensity estimate
```

For each surviving streamline, TIDE identifies the AF magnitude required to sustain activation over a configured contiguous length. The primary cross-streamline summary is the **median of the top 5%** of the resulting per-streamline threshold distribution. Optional SIFT2 weights produce a weighted counterpart.

If `M_CST` and `M_target` are the corresponding bundle metrics, the raw target estimate is:

```text
I_TIDE,raw = RMT × (M_CST / M_target)
```

TIDE also reports the **Stimulation Efficiency Index**:

```text
SEI = M_target / M_CST
```

`SEI > 1` indicates that the target pathway is more efficient than the CST under the simulated configuration; `SEI < 1` indicates lower efficiency.

The raw estimate is retained in the outputs. A configurable intensity clamp is additionally reported for QC/operational use; by default it is bounded relative to RMT and by the device maximum. Clamp status is explicit (`WITHIN_RANGE`, `CLAMPED_LOW`, `CLAMPED_HIGH`, or `DEVICE_LIMITED`).

---

## Getting Started

### Prerequisites

Before running TIDE you need:

- **SimNIBS** installed separately (the reference development environment uses SimNIBS 4.5);
- **Python 3.11–3.12** for the TIDE package/launcher; the reference computational runtime is the Python bundled with **SimNIBS 4.5 (Python 3.11)**;
- a subject-specific SimNIBS `m2m_*` head model containing a single `.msh` head mesh;
- the subject's T1-weighted anatomical image;
- a CST tractogram for motor calibration (`.trk`, loaded in RASMM space);
- a target tractogram (`.trk`);
- the measured motor threshold in `%` maximum stimulator output;
- a SimNIBS-compatible TMS coil model and the device maximum `dI/dt`.

Optional inputs include SIFT2 weights, a FreeSurfer surface in scanner RAS, and a Softaxic STMPX template.

> [!TIP]
> Generate a fresh annotated configuration anywhere with `tide --init-config config.yml`. The same canonical template is also available as [`config_template.yml`](https://github.com/marcotag93/TIDE/blob/main/config_template.yml) in the repository.

### Naming: repository, package, import, and command

TIDE intentionally uses different names for different distribution layers:


| Layer                                 | Name            |
| --------------------------------------- | ----------------- |
| Research software / GitHub repository | **TIDE**        |
| PyPI distribution                     | `tide-pipeline` |
| Python import package                 | `tide`          |
| Command-line entry point              | `tide`          |

This means that users install the **distribution** as `tide-pipeline` but run the software with the shorter `tide` command.

> [!WARNING]
> The PyPI project named `tide` is an unrelated package. Do **not** use `pip install tide` to install this software; use `tide-pipeline`.

### 1. Clone the repository (source installation only)

Skip this step when installing the published package from PyPI.

```bash
git clone https://github.com/marcotag93/TIDE.git
cd TIDE
```

### 2. Install TIDE

#### Recommended: PyPI installation

TIDE is distributed on PyPI as [`tide-pipeline`](https://pypi.org/project/tide-pipeline/). The recommended approach for normal users is to keep the initial TIDE launcher isolated from existing scientific Python environments, then explicitly bootstrap the same published release into SimNIBS. With [`pipx`](https://pipx.pypa.io/):

```bash
pipx install --python 3.11 tide-pipeline
tide --bootstrap
```

Or with [`uv`](https://docs.astral.sh/uv/guides/tools/):

```bash
uv tool install --python 3.11 tide-pipeline
tide --bootstrap
```

This avoids resolving TIDE's pinned numerical dependencies directly into an existing FSL, Conda, system-Python, or other research environment. `tide --bootstrap` then locates the SimNIBS Python explicitly, verifies its numerics-critical versions against TIDE's pins, and installs the matching non-editable `tide-pipeline` release there.

A conventional interpreter-specific installation is also supported when you deliberately want TIDE in that Python:

```bash
python -m pip install tide-pipeline
python -m tide --bootstrap
```

Using `python -m tide --bootstrap` guarantees that the bootstrap is executed by the same interpreter into which `tide-pipeline` was just installed. The bootstrap aborts on a SimNIBS dependency mismatch unless `--force` is explicitly supplied.

After a successful bootstrap:

```bash
tide --help
```

If more than one `tide` executable exists on `PATH`, use the SimNIBS interpreter explicitly or inspect the selected launcher with `type -a tide` / `command -v tide`.

> [!WARNING]
> SimNIBS is a **system dependency** and is intentionally not installed from PyPI by TIDE. The computational workflows must execute under the SimNIBS Python environment.

#### Recommended source installation

For development or direct use of a source checkout, the recommended installation path is the bundled SimNIBS-aware installer:

```bash
python install.py --simnibs-env --editable
```

This command is intentionally safe to launch even from another Python environment (for example FSL or a system Python): `install.py` uses only the standard library to locate the SimNIBS installation, selects the SimNIBS Python explicitly, verifies the numerics-critical dependency versions against TIDE's pins, installs TIDE into that environment, and checks that `import tide` succeeds with the same interpreter.

After installation:

```bash
tide --help
```

If your shell has multiple `tide` launchers on `PATH`, the interpreter-explicit form is always unambiguous:

```bash
/path/to/SimNIBS/simnibs_env/bin/python -m tide --help
```

If you already know the exact SimNIBS interpreter and have independently verified its dependency versions, you may install directly with it:

```bash
/path/to/SimNIBS/simnibs_env/bin/python -m pip install -e .
```

#### Advanced: install into the current Python environment

A standard pip install remains supported:

```bash
python -m pip install .
```

This installs TIDE into **that exact Python interpreter**. It does not automatically redirect the installation into SimNIBS. This mode is useful for development, packaging checks, or for installing a temporary launcher that will subsequently bootstrap TIDE into SimNIBS. For normal source-based pipeline use, prefer `python install.py --simnibs-env --editable`.

If you deliberately use the current-environment route, verify it before invoking a console script:

```bash
python -c "import tide; print(tide.__version__, tide.__file__)"
python -m tide --help
```

> [!IMPORTANT]
> Prefer `python -m pip` over a bare `pip` command. A bare `pip`, `python`, and `tide` can each resolve to different environments on neuroimaging workstations that expose FSL, SimNIBS, Conda, system Python, or user-local executables on the same `PATH`.

#### Troubleshooting: `ModuleNotFoundError: No module named 'tide'`

If the `tide` executable exists but immediately fails with `ModuleNotFoundError`, the most common cause is that the console script and the installed package come from different Python environments, or that the executable is stale from an older/failed installation. Diagnose the active command first:

```bash
type -a python python3 pip tide
pip --version
head -n 1 "$(command -v tide)"
python -m pip show tide-pipeline
python -c "import sys, site; print(sys.executable); print(site.getusersitepackages())"
```

For a source checkout, the most reliable repair is to install directly with the SimNIBS interpreter and then invoke the command from the same environment:

```bash
/path/to/SimNIBS/simnibs_env/bin/python -m pip install -e .
/path/to/SimNIBS/simnibs_env/bin/python -c "import tide; print(tide.__version__, tide.__file__)"
/path/to/SimNIBS/simnibs_env/bin/python -m tide --help
/path/to/SimNIBS/simnibs_env/bin/tide --help
```

If `command -v tide` still resolves to an older `~/.local/bin/tide`, but the interpreter-explicit `python -m tide --help` command works, the installation itself is healthy and the problem is only command resolution. Refresh Bash's command cache (`hash -r`), remove/uninstall the stale launcher from the Python environment that created it, or place the intended environment's `bin` directory before `~/.local/bin` on `PATH`. Do not copy a launcher manually between Python environments: console scripts are tied to the interpreter that generated them.

### 3. Create a configuration

Generate the complete annotated template from any installation:

```bash
tide --init-config config.yml
```

If no output path is supplied, TIDE writes `./config.yml`:

```bash
tide --init-config
```

For safety, `--init-config` never overwrites an existing file. Replace every `/path/to/...` placeholder with an absolute path and edit the subject, coil, calibration, and target sections for your experiment.

### 4. Run an estimation

```bash
tide --config config.yml --workflow estimation
```

A successful CLI run exits with status `0` and prints `PIPELINE COMPLETE` only after the required workflow outputs have been written.

---

## Configuration

TIDE uses a YAML configuration file. Run `tide --init-config config.yml` to materialize the complete annotated reference from the installed package; the same canonical file is available as [`config_template.yml`](https://github.com/marcotag93/TIDE/blob/main/config_template.yml). The example below shows only the core fields required to understand a standard estimation run.

```yaml
subject:
  id: "sub-001"
  derivatives_path: /absolute/path/to/derivatives/sub-001
  m2m_path: /absolute/path/to/m2m_sub-001
  files:
    t1w: /absolute/path/to/sub-001_T1w.nii.gz
    # weights_cst: /absolute/path/to/CST_weights.txt
    # weights_target: /absolute/path/to/target_weights.txt
    # surface: /absolute/path/to/lh.white.scanner.white

workflow: estimation

coil:
  coil_model: "MagVenture_C-B60.ccd"
  coil_path: ""                 # empty = auto-detect SimNIBS coil directory
  coil_distance_mm: 4.0
  device_didt_max: 161e6        # A/s; set this for your stimulator/coil

options:
  roi_size_mm: 20.0
  activation_length_mm: 6.0
  field_mode: "af"             # required for estimation and grid workflows
  gwi_threshold_mm: 3.0
  adm_optimization: true
  mso_floor_ratio: 0.70
  mso_ceiling_ratio: 1.40
  generate_visualizations: true
  generate_3d_visualization: false

experiment:
  calibration:
    label: "M1"
    bundle_path: /absolute/path/to/CST_left.trk
    coords: [-13.28, -26.71, 63.0]
    scalp_coords: [-13.28, -26.71, 85.0]
    orientation: "C3"          # or [x, y, z] or a rigid 4x4 matsimnibs matrix
    measured_rmt_mso: 38.0

  target:
    label: "TARGET"
    bundle_path: /absolute/path/to/target_bundle.trk
    coords: [-40.0, 35.0, 30.0]
    scalp_coords: [-60.0, 35.0, 30.0]
    orientation: "F3"          # or [x, y, z] or a rigid 4x4 matrix
    cortical_medoid: false
```

### Orientation priority

For calibration and target sites, `orientation` accepts:

1. **Rigid 4×4 `matsimnibs` matrix** — used directly; optimisation is skipped.
2. **Three-coordinate vector** `[x, y, z]` — used as the SimNIBS `pos_ydir` reference.
3. **EEG 10–20 label** such as `"F3"` or `"F8"`.

For `--workflow grid`, the target orientation must be a vector or EEG label, **not** a 4×4 matrix, because each grid point is independently optimised from the supplied seed.

### Important configuration rules

- `field_mode: "af"` is required for `estimation` and `grid`.
- Use absolute paths wherever possible.
- Tractograms are loaded in **RASMM** space using the T1w image as anatomical reference.
- A configured weight or surface file is treated as an explicit input: invalid/missing files fail rather than silently falling back.
- The final coil pose is checked for geometric QC before dose estimation.
- A saved estimation configuration contains the resolved final matrices and can be replayed without re-running optimisation.

---

## Workflows


| Workflow         | Command                                            | Purpose                                                                                 |
| ------------------ | ---------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| **Estimation**   | `tide --config config.yml --workflow estimation`   | Full CST-calibrated TIDE intensity estimation for one target.                           |
| **Grid search**  | `tide --config config.yml --workflow grid`         | Evaluate a set of candidate target positions and produce a spatial intensity map.       |
| **Simulation**   | `tide --config config.yml --workflow simulation`   | Run a standard TMS E-field simulation; can map AF or`e_parallel` along a target bundle. |
| **Optimization** | `tide --config config.yml --workflow optimization` | Run coil-position optimisation only.                                                    |

### Estimation

```bash
tide --config config.yml --workflow estimation
```

The estimation workflow processes M1/CST and target branches in parallel where possible, then combines their bundle metrics to produce weighted and unweighted intensity estimates, SEI, QC fields, and a reproducibility configuration.

### Grid search

Add a nested grid block under `experiment.target`:

```yaml
experiment:
  target:
    label: "TARGET"
    bundle_path: /absolute/path/to/target_bundle.trk
    coords: [-40.0, 35.0, 30.0]
    scalp_coords: [-60.0, 35.0, 30.0]
    orientation: "F3"
    grid:
      search_radius_mm: 20.0
      step_size_mm: 4.0
      cortex_depth_mm: 2.0
```

Then run:

```bash
tide --config config.yml --workflow grid
```

Each successful grid point receives its own replayable estimation configuration containing the resolved coordinates, scalp position, and final 4×4 target pose.

### Standard simulation

```bash
tide --config config.yml --workflow simulation
```

Use this workflow when you need a conventional simulation or bundle field mapping without forming the CST-to-target dose ratio. `field_mode: "e_parallel"` is supported here.

### Standard optimisation

```bash
tide --config config.yml --workflow optimization
```

This workflow returns the optimised coil pose without running the full TIDE dose-estimation pipeline.

---

## Python API

The `tide` package can also be used programmatically. For most applications, the recommended API is to load the same YAML configuration used by the CLI and call a workflow entry point directly.

### Run a workflow from Python

```python
from tide.utils.config import SimNIBSConfig
from tide.workflows.estimation import run_estimation_workflow

config = SimNIBSConfig.from_yaml("config.yml")
run_estimation_workflow(config, console_ui=False)
```

The main workflow entry points are:

```python
from tide.workflows.estimation import run_estimation_workflow
from tide.workflows.grid_search import run_grid_search_workflow
from tide.workflows.standard import run_standard_optimization, run_standard_simulation
```

Each accepts a `SimNIBSConfig` loaded with `SimNIBSConfig.from_yaml(...)`. For library use, you can explicitly call `validate_workflow_config(config, workflow)` before dispatching a workflow; the estimation, grid-search, and simulation entry points also perform their own workflow validation.

For applications that need progress reporting during a grid search, `run_grid_search_workflow()` accepts a callback:

```python
from tide.utils.config import SimNIBSConfig
from tide.workflows.grid_search import run_grid_search_workflow

config = SimNIBSConfig.from_yaml("config.yml")


def on_progress(completed: int, total: int, label: str) -> None:
    print(f"{completed}/{total}: {label}")


run_grid_search_workflow(
    config,
    progress_callback=on_progress,
    console_ui=False,
)
```

### Lower-level scientific functions

Advanced users can access the numerical building blocks directly. Two useful entry points are:

```python
from tide.core.physics import calculate_scalar_map
from tide.interfaces.unified_estimation import run_unified_estimation
```

- `calculate_scalar_map(...)` computes signed activating-function or `E_parallel` values along streamlines.
- `run_unified_estimation(...)` performs CST-to-target intensity estimation from already generated AF tractograms, with optional SIFT2 weights and grey-white-interface surface constraints.

> [!NOTE]
> The workflow API is the preferred programmatic interface because it applies TIDE's configuration preflight, SimNIBS orchestration, output handling, and workflow-level safety checks. Lower-level functions are intended for custom analyses by users who understand their input and unit contracts.

---

## Outputs

### Estimation

A typical estimation run writes:

```text
<derivatives_path>/TIDE_<target>/
├── TIDE_Results_<target>.txt
├── TIDE_Results_<target>.json
├── TIDE_Results_<target>.html
├── config_estimation_*.yml
├── sim_m1/
│   ├── CST_M1_af.trk
│   └── CST_M1_af.nii.gz              # when visualisation is enabled
├── sim_target/
│   ├── <target>_af.trk
│   └── <target>_af.nii.gz            # when visualisation is enabled
└── visualizations/                    # optional figures / interactive renders
```

The report includes, among other fields:

- raw and clamped weighted/unweighted intensity estimates;
- clamp/QC status;
- CST and target bundle AF metrics;
- weighted and unweighted SEI;
- intensity multipliers;
- coil matrices and pose QC;
- alignment/depth diagnostics;
- aggregator-sensitivity diagnostics; and
- provenance/configuration information.

### Grid search

A grid run additionally produces:

```text
<derivatives_path>/TIDE_grid_search_<...>/
├── TIDE_grid_results.csv
├── TIDE_Grid_Summary_<target>.txt
├── TIDE_Grid_Summary_<target>.json
├── TIDE_Grid_Summary_<target>.html
├── calibration_m1/
├── simulations/
│   ├── grid_P01/
│   ├── grid_P02/
│   └── ...
├── QC/
└── visualization/
    ├── grid_mso_raw_map.nii.gz
    ├── grid_mso_map.nii.gz
    ├── grid_mso_flag_map.nii.gz
    └── grid_interactive.html          # when 3D visualisation is enabled
```

Historical machine-readable names containing `mso` are intentionally retained for backwards compatibility even though human-facing reports use intensity notation.

---

## Advanced Usage

### SIFT2 weighting

Provide one weight per original streamline:

```yaml
subject:
  files:
    weights_cst: /absolute/path/to/CST_weights.txt
    weights_target: /absolute/path/to/target_weights.txt
```

TIDE tracks original streamline identities through filtering/dropping so surviving streamlines remain aligned with their corresponding weights.

### Surface-constrained analysis

Provide a FreeSurfer surface in scanner RAS:

```yaml
subject:
  files:
    surface: /absolute/path/to/lh.white.scanner.white

options:
  gwi_threshold_mm: 3.0
```

The surface constraint is applied only when the surface is explicitly configured.

### Softaxic STMPX export

After a successful estimation:

```bash
tide --config config.yml --workflow estimation --stmpx /path/to/session.stmpx
```

TIDE validates the template before the workflow starts and writes:

```text
/path/to/session_updated.stmpx
```

The STMPX file is an **output template**, not a source of simulation pose parameters. The YAML configuration remains authoritative for the SimNIBS input pose.

### Fixed-pose cache

TIDE caches deterministic SimNIBS artifacts for exact 4×4 fixed poses. The default cache root is:

```text
$XDG_CACHE_HOME/tide/fixed_pose
```

or, when `XDG_CACHE_HOME` is not set:

```text
~/.cache/tide/fixed_pose
```

Useful commands:

```bash
tide --cache-info
tide --cache-clear
tide --config config.yml --workflow estimation --no-cache
```

Configuration equivalents:

```yaml
subject:
  cache_dir: /absolute/path/to/tide_cache  # relocate
  cache_max_size_gb: 100                   # optional LRU cap; 0/omitted = unlimited
```

Set `cache_dir: no` to disable the fixed-pose cache from YAML.

### Console and logging

```bash
# Standard output
tide --config config.yml --workflow estimation --verbosity standard

# More detail
tide --config config.yml --workflow estimation --verbosity verbose

# Minimal output
tide --config config.yml --workflow estimation --verbosity quiet

# Disable the rich grid console UI
tide --config config.yml --workflow grid --no-console-ui
```

### Optional 3D visualisation

3D rendering is an optional dependency:

```bash
python -m pip install ".[viz]"
```

Enable it in YAML:

```yaml
options:
  generate_3d_visualization: true
```

---

## For Developers

### Project structure

```text
TIDE/
├── main.py                       # source-checkout CLI shim
├── config_template.yml           # canonical annotated configuration template
├── install.py                    # installation convenience wrapper
├── pyproject.toml                # package metadata and pinned direct dependencies
├── uv.lock                       # frozen development/reproduction environment
├── src/tide/
│   ├── cli.py                    # `tide` console entry point + --init-config
│   ├── core/                     # AF, geometry, tractography, scientific I/O
│   ├── interfaces/               # SimNIBS, sampling, estimation, visualisation, STMPX
│   ├── workflows/                # estimation, grid, simulation, optimisation
│   ├── console/                  # rich terminal UI / worker reporting
│   └── utils/                    # config, logging, cache, SimNIBS discovery
├── tests/                        # pytest suite
└── .github/workflows/            # CI and release automation
```

### Development install

```bash
python -m pip install -e ".[dev]"
```

Run the checks used by CI:

```bash
black --check --diff .
isort --check-only --diff .
flake8 . --select=E9,F63,F7,F82 --show-source --statistics
mypy src/tide
pytest tests/ --cov=tide --cov-branch --cov-fail-under=45
```

SimNIBS-dependent end-to-end runs should be executed in a separate scratch output directory and must never overwrite reference derivatives.

The build backend is pinned to **Hatchling 1.27.0** in `pyproject.toml`, matching TIDE's reproducibility-oriented policy of pinning the software versions that define its tested packaging and numerical environment.

### Frozen environment

`pip install tide-pipeline` resolves the direct dependency pins declared in `pyproject.toml`; it does not consume `uv.lock`. To reproduce the complete tested development environment from a checkout, use:

```bash
uv sync --locked --extra dev
```

Each GitHub Release also carries the exact `uv.lock` used for that release, the wheel and source distribution, and `SHA256SUMS`. SimNIBS remains a separately installed system dependency; the reference computational runtime is SimNIBS 4.5 with Python 3.11.

---

## License

TIDE is released under the **GNU General Public License v3.0 or later (GPL-3.0-or-later)**. See [`LICENSE`](https://github.com/marcotag93/TIDE/blob/main/LICENSE) for the full license text.

---

## Acknowledgments

TIDE builds on open scientific software, including:

- [SimNIBS](https://simnibs.github.io/simnibs/) for finite-element TMS modelling and coil optimisation;
- [DIPY](https://dipy.org/) for tractography I/O and geometric processing;
- [NiBabel](https://nipy.org/nibabel/) for neuroimaging I/O;
- [SciPy](https://scipy.org/) and [NumPy](https://numpy.org/) for numerical computing; and
- [PyVista](https://pyvista.org/) / [VTK](https://vtk.org/) for optional 3D visualisation.

---

## Contact

For scientific or software questions, bug reports, or feature requests, please use the repository's GitHub Issues page or contact:

- **Academic email:** [marco.tagliaferri@unitn.it](mailto:marco.tagliaferri@unitn.it)
- **Permanent email:** [marco.tagliaferri93@gmail.com](mailto:marco.tagliaferri93@gmail.com)

---

<p align="center">
  Made with ❤️ for the TMS research community
</p>
