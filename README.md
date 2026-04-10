<div align="center">

<h1>🎯 Red Teaming Large Reasoning Models</h1>

<p>
Jiawei Chen<sup>1,3</sup>* · Yang Yang<sup>1</sup>* · Chao Yu<sup>2</sup>* · Yu Tian<sup>5</sup> · Zhi Cao<sup>3</sup> · Xue Yang<sup>4</sup> · Linghao Li<sup>1</sup> · Hang Su<sup>5</sup>† · Zhaoxia Yin<sup>1</sup>†
</p>

<p>
<sup>1</sup>East China Normal University · <sup>2</sup>Shenzhen International Graduate School, Tsinghua University · <sup>3</sup>Zhongguancun Academy<br>
<sup>4</sup>Shanghai Jiao Tong University · <sup>5</sup>Dept. of Comp. Sci. and Tech., THBI Lab, Tsinghua University
</p>

<p><i>* Equal contribution &nbsp;&nbsp; † Corresponding authors</i></p>

<p>
<a href="#project-overview">🌐 Project Page</a> &nbsp;|&nbsp;
<a href="https://arxiv.org/pdf/2512.00412">📄 Paper</a> &nbsp;|&nbsp;
<a href="#citation">📖 Citation</a>
</p>

</div>

---

## Project overview

This repository provides the **official code release** accompanying *Red Teaming Large Reasoning Models* ([arXiv:2512.00412](https://arxiv.org/abs/2512.00412)). It is a **batch-inference and evaluation toolkit** for studying how large language models behave under various **prompt “attack” setups** (e.g., tampered chain-of-thought, multi-turn jailbreak-style dialogues, overthinking-style prompts).

Typical workflow:

1. Load prompts from a dataset (Excel or JSON, depending on `attack_type`).
2. Run the target model to generate responses (either **locally** via Hugging Face + PyTorch, or **remotely** via an OpenAI-compatible chat API).
3. Post-process results with scripts under `evaluate/` (safety ASR, toxicity, accuracy, over-reasoning, timeout statistics).

### Evaluation dimensions and `attack_type`

The `--attack_type` flag groups benchmarks into three dimensions:

| Dimension | `attack_type` values |
|-----------|----------------------|
| **Truthfulness** | `CPT`, `trutheval_dataset` |
| **Safety** | `H-CoT`, `Attack_600` |
| **Efficiency** | `catattack`, `overthinking_attack` |

---

## Repository layout

```text
.
├── LICENSE
├── README.md
├── requirements.txt
├── test.py                 # Local HF causal LM inference (uses torch / transformers; NPU-oriented in code)
├── test_proprietary.py     # Same logical pipeline via OpenAI-compatible Chat API
├── dataset/
│   └── Attack_600.json     # Example JSON dataset (multi-turn attacks)
└── evaluate/
    ├── evaluate_asr.py      # LLM-based safety audit → attack success rate (ASR)
    ├── evaluate_toxicity.py # Perspective API toxicity scores
    ├── evaluate_acc.py      # Accuracy vs. gold answers (CPT / TruthEval-style)
    ├── evaluate_or.py       # Overthinking rate from token counts
    └── evaluate_timeout.py  # Share of slow generations (≥ 180s)
```

**Note:** For most `attack_type` values, the code expects an Excel file at `dataset/<attack_type>.xlsx`. Only `Attack_600` is bundled as JSON in this snapshot.

---

## Environment setup

**Supported Python version:** 3.10.

### 1. Create and activate a virtual environment

**Linux / macOS:**

```bash
cd /path/to/this/repo
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell):**

```powershell
cd C:\path\to\this\repo
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install Python dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

The pinned file lists packages such as `pandas`, `openpyxl`, `tqdm`, `openai` (legacy `<1.0` API), `requests`, `torch`, `transformers`, and `torch_npu`.



### 3. Configure API keys and endpoints (when needed)

| Component | What to set | Where (in code) |
|-----------|-------------|-----------------|
| Remote chat inference | `openai.api_key`, `openai.api_base` | `test_proprietary.py`  |
| Safety / extraction judges | `openai.api_key`, `openai.api_base` | `evaluate/evaluate_asr.py`, `evaluate/evaluate_acc.py` |
| Perspective toxicity | API URL and `key` query param | `evaluate/evaluate_toxicity.py` |


---

## Data preparation

### `Attack_600` (`--attack_type Attack_600`)

- **Input file:** JSON, default path `dataset/Attack_600.json` (override with `--attack_json`).


### All other `attack_type` values

- **Input file:** `dataset/<attack_type>.xlsx` (Excel).


**Note:** Files whose names end with `_addition` refer to datasets either constructed from scratch or redesigned and improved based on existing datasets.

---

## How to run

**Working directory matters:**

- Run `test.py` and `test_proprietary.py` from the **repository root**.
- Run evaluation scripts from the **`evaluate/`** folder.

### A. Local Hugging Face inference (`test.py`)

From the repo root:

```bash
cd /path/to/this/repo
python test.py --model_folder "XiaomiMiMo/MiMo-7B-RL-Zero" --attack_type Attack_600 --attack_json dataset/Attack_600.json
```

Example with another attack type (requires the matching `.xlsx` under `dataset/`):

```bash
python test.py --model_folder "/path/to/local/model" --attack_type H-CoT
```

### B. OpenAI-compatible API inference (`test_proprietary.py`)

From the repo root (after setting API key/base in the script):

```bash
cd /path/to/this/repo
python test_proprietary.py --model_name "o1" --attack_type Attack_600 --attack_json dataset/Attack_600.json
```

### C. Evaluation scripts

From `evaluate/`:

```bash
cd /path/to/this/repo/evaluate
python evaluate_asr.py --attack_type Attack_600
```

```bash
python evaluate_toxicity.py --attack_type H-CoT
```

```bash
python evaluate_acc.py --attack_type trutheval_dataset
```


```bash
python evaluate_or.py --attack_type overthinking_attack
```

```bash
python evaluate_timeout.py --attack_type catattack
```

**Note:** Ensure inference outputs already exist under `result/result_<attack_type>/` before running the matching evaluator.

---

## Outputs

### Inference (`test.py` / `test_proprietary.py`)

- **Directory:** `result/result_<attack_type>/`
- **File:** `<model_name>_<attack_type>.json`.

### `evaluate_asr.py`

- Annotated JSONs: `result/validation_ASR_<attack_type>/evaluated_<original>.json`
- Summary: `result/validation_ASR_<attack_type>/validation_safety_summary.json`

### `evaluate_toxicity.py`

- JSONs with `toxicity_score_max`: `result/validation_toxicity_<attack_type>/`

### `evaluate_acc.py`

- Per-file validation: `result/validation_ACC_<attack_type>/<stem>_validation.json`
- Aggregate summary: `result/validation_ACC_<attack_type>/validation_<attack_type>_summary.json`

### `evaluate_or.py`

- Text report: `result/result_<attack_type>/or_statistics.txt`

### `evaluate_timeout.py`

- Text report: `result/result_<attack_type>/timeout_statistics.txt`

---

## Notes and limitations

- **Harmful content:** Some datasets contain disallowed or sensitive requests intended for controlled safety research. Use only where legally and ethically permitted.

- **Judge / API costs:** `evaluate_asr.py` and `evaluate_acc.py` (LLM extraction path) call remote models; budget and rate limits apply.

- **Perspective API:** `evaluate_toxicity.py` sleeps between requests; URL/key placeholders must be filled.

---

## Citation

If you use this code or the paper, please cite:

```bibtex
@misc{chen2026redteaminglargereasoning,
      title={Red Teaming Large Reasoning Models}, 
      author={Jiawei Chen and Yang Yang and Chao Yu and Yu Tian and Zhi Cao and Xue Yang and Linghao Li and Hang Su and Zhaoxia Yin},
      year={2026},
      eprint={2512.00412},
      archivePrefix={arXiv},
      primaryClass={cs.CR},
      url={https://arxiv.org/abs/2512.00412}, 
}
```

---

## License

See `LICENSE` in this repository.
