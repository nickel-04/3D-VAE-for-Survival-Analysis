# 3D VAE for Survival Analysis

Survival prediction from 3D CT scans using a Variational Autoencoder. This adapts **SAVAE** ([Apellániz et al., 2023](https://arxiv.org/abs/2312.14651)) from clinical data to 3D images of patients with colorectal liver metastases.

Course project for IN9310 – Advanced Deep Learning for Image Analysis, University of Oslo.

## Branches

| Branch | Contents |
|---|---|
| `main` | Code, data and a Makefile that sets up the environment and trains the models |
| `report` | LaTeX report and a CI/CD pipeline that builds the report PDF |

## `main`: train the model

```bash
git clone https://github.com/nickel-04/3D-VAE-for-Survival-Analysis.git
cd 3D-VAE-for-Survival-Analysis
make
```

The **Makefile** runs the whole pipeline with one `make` command: it sets up the environment, downloads the [CT data](https://www.cancerimagingarchive.net/collection/colorectal-liver-metastases) and trains all models. Run a single step with e.g. `make exp1`.

| Experiment | Survival head |
|---|---|
| `exp1`, `exp1.1` | Weibull distribution (parametric) |
| `exp2`, `exp2.1` | Survival curve at 10 time points (non-parametric) |

Plots and model checkpoints are saved to `output/`. The best result was from Experiment 1: **C-index 0.62, IBS 0.19**.

## `report`: CI/CD for the report

Every push to `report` runs a GitHub Actions workflow that compiles `report.tex` with LaTeX and uploads the PDF as an artifact. Pushing a tag also attaches the PDF to a GitHub Release.

- **Get the PDF:** Actions tab → latest run → Artifacts
- **Make a release:** `git tag v1.0 && git push origin v1.0`
- **Setup:** needs a self-hosted runner and repository variables `ENGINE` (e.g. `pdflatex`) and `FILE` (`report`)
