# Perception-LLM-Analysis

A research-oriented pipeline for roadside perception and defect analysis using multi-modal sensor data, object detection, and a two-phase LLM verification workflow.

This project combines video, GPS, IMU, and road-context signals to detect surface defects and verify them using a staged reasoning pipeline. It is designed for structured evaluation, reproducible runs, and experimental analysis across multiple traces.

## Overview

The repository contains a full workflow for:

- extracting and synchronizing dashcam, GPS, and sensor data
- running multimodal detection and context construction
- evaluating road-quality hypotheses with rule-based and LLM-based verification
- comparing pipeline variants and reporting metrics across traces
- supporting research iterations on perception, federated learning, and verification logic

## Project goals

- Build a robust perception pipeline over PVS-style driving traces
- Combine visual detection with contextual road metadata
- Use a staged LLM verification process to reduce false positives
- Enable trace-wise analysis and evaluation for research reporting
- Keep the project reproducible, organized, and easy to extend

## Architecture at a glance

- Detection: YOLO-based object workflows
- Context: GPS, IMU, and road metadata enrichment
- Reasoning: text-only prior + multimodal verification stage
- Evaluation: metrics, ablations, and result summaries
- Infrastructure: local pipeline orchestration and experiment scripts

## Repository structure

```text
.
├── README.md
├── START_HERE.md
├── WINDOWS_SETUP.md
├── REQUIREMENTS.md
├── FINAL.md
├── FIX.md
├── RUN.md
├── STEPS.md
├── config/
├── configs/
├── chain/
├── data/
├── Dataset/
├── notebooks/
├── perception/
├── results/
├── runs/
├── src/
├── weights/
├── .gitignore
├── .env.example
├── requirements.txt
├── check.bat
├── smoke.bat
├── run_full.bat
├── run_all_traces.bat
└── setup.bat
```

## Tech stack

- Python 3.11+
- PyTorch + CUDA 12.8 support
- Ultralytics / YOLO-based detection
- GPS + sensor processing
- Overpass / road-context enrichment
- LLM verification flow using text and multimodal stages
- Bash / Windows batch automation for experiment execution

## Prerequisites

Before running the pipeline, ensure the following are available:

- Python 3.11
- NVIDIA driver compatible with CUDA 12.8
- ffmpeg and ffprobe
- Git
- A valid model directory under `weights/`
- Environment variables configured in `.env`

## Quick start

1. Clone the repository
2. Create a virtual environment
3. Install dependencies
4. Copy `.env.example` to `.env` and fill in your credentials and dataset paths
5. Run the smoke test
6. Validate synchronization before a full run

Example:

```bash
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
python -m src.pipeline.run --side left --smoke
```

On Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
python -m src.pipeline.run --side left --smoke
```

## Environment configuration

The project expects a `.env` file containing keys like:

```env
DASHSCOPE_API_KEY=your_api_key
OVERPASS_CONTACT=you@example.com
DATASET_DIR=C:/path/to/Dataset
DEFAULT_TRACE=PVS 2
```

Keep secrets out of source control. The project uses `.gitignore` to avoid committing local credentials and generated artifacts.

## Typical workflow

### 1. Setup and validation

```bash
python -m src.utils.check_setup
```

### 2. Smoke test

```bash
python -m src.pipeline.run --side left --smoke
```

### 3. Full run

```bash
python -m src.pipeline.run --side left
```

### 4. Evaluation

```bash
python -m src.eval.make_eval_set --side left --n 300
python -m src.eval.metrics --side left
```

## Research notes

This project is intended for experimental and research usage. Several places in the pipeline include tunable thresholds or heuristic assumptions that should be validated before publication, including:

- fusion weights
- phase gate thresholds
- extraction FPS settings
- synchronization assumptions between video and sensor streams

These are documented in the project notes and should be revisited during evaluation and reporting.

## Key documentation

- [START_HERE.md](START_HERE.md) — primary local setup and execution guide
- [WINDOWS_SETUP.md](WINDOWS_SETUP.md) — Windows-specific setup notes
- [RUN.md](RUN.md) — execution and pipeline reference
- [FINAL.md](FINAL.md) — project summary and analysis notes
- [STEPS.md](STEPS.md) — development workflow and task sequencing

## Contributing

Contributions are welcome in the form of bug fixes, new evaluation logic, performance improvements, documentation updates, and research extensions.

## License

This project is released under the MIT License unless otherwise stated.

## Acknowledgements

This repository is built for research experimentation around perception-based road defect detection and verification. It is intended to support iterative model evaluation and reproducible experimental workflows.

---

If you are getting started locally, begin with [START_HERE.md](START_HERE.md).