# Where Memorized-Secret Extraction Fails: A Defense-Layer

Measurement of Canary Leakage in Protected Multimodal Large Language Models



Multimodal large language models (MLLMs) fine-tuned on private data can

memorize secrets. Deployments usually protect them with two layers: a

confidentiality **system prompt** and an **output filter**. This repository

measures the two layers separately:



**Implantation.** Secrets (canaries) are implanted into Qwen2-VL-7B and
   LLaVA-1.5-7B with LoRA fine-tuning.

2. **Qualification.** A target enters the study only if it passes a

   memorization gate (every implanted secret is recovered exactly, no

   never-implanted secret is) and a utility check.

3. **Four-level defense ladder.** Every query is answered with a neutral and

   with the protected system prompt, and every answer is scored before and

   after the filter. The four levels (N = no defense, S = system prompt only,

   F = filter only, P = both) therefore share the same generations.

4. **Attacks.** Fixed prompts (Direct, DAN, Paraphrase), visual attacks

   (FigStep, form/HADES-style images), adaptive attackers (PAIR,

   AutoDAN-Turbo-style, HADES black-box) and the MM-LEAK PPO attacker are run

   on the same identities and budgets.

<img width="2064" height="1392" alt="overview_measurement before_review_20261023" src="https://github.com/user-attachments/assets/6ae7d8c9-4620-4f94-b1f7-c0f3ded0e9cb" />




The study then asks why the filter works (recall is literal), which

disclosures it misses (untagged and partial copies), and where it fails (raw

answers kept in the conversation history). 
## Repository layout

```
script/scripts/            experiment code and one launcher per experiment (launch_*.sh)
script/configs/            frozen protocol (revision_v1.json) and attacker strategy banks
script/assets/figstep/     font used to render FigStep images
script/paper/figures/      builders of the paper figures and tables from the reports
script/script_new/       shared fixed-prompt templates and feedback of the attacker
Dataset/                 public replay text (Wikipedia / C4) and the synthetic utility tasks
```

Scripts resolve every path from the repository root, so keep this layout.
Generated data go to `TDSC/<experiment>/` (public, key-free case files),
`private/` (evaluator keys, training sets, raw responses, reports) and
`runs/` (adapters). `private/` and `runs/` are git-ignored and must never be
shared with an attacker process.

## Setup

Python 3.10, CUDA GPUs with at least 20 GB free (30 GB for LLaVA training).

```bash
conda create -n mmleak python=3.10 -y
conda activate mmleak
pip install -r requirements.txt
```

The launchers use `python3` from the active environment and the repository
root by default; override with `MMLEAK_PYTHON` and `MMLEAK_ROOT`. Select the
GPU with `GPU_ID` (default 0). Every launcher is resumable and stops at a
failed gate. Base models (`Qwen/Qwen2-VL-7B-Instruct`,
`llava-hf/llava-1.5-7b-hf`) are downloaded from the Hugging Face Hub.

## Reproducing the study

Run the commands from the repository root. Each step reads the outputs of the
previous ones.

### 1. Targets

| Step | Command | Output |
|---|---|---|
| Qwen2-VL targets 101–105 (secret generation, LoRA dev/eval, memorization gate, MM-LEAK PPO) | `python TDSC/scripts/revision_v1.py plan --gpu-ids 0 1` then `python TDSC/scripts/revision_v1.py run --python "$(command -v python3)" --gpu-ids 0 1` | `private/revision_v1`, `runs/revision_v1` |
| LLaVA-1.5 targets 101–103 (LoRA on the same secrets, gate, utility) | `bash TDSC/scripts/launch_llava_v3_seeds_train.sh`, then `bash TDSC/scripts/launch_llava_v3_seeds_measure.sh` | `runs/revision_v1_llava_v3_lr2e5` |

`revision_v1.py` calls `prepare_revision_v1.py` (secrets and data),
`finetune_qwen2vl.py` (LoRA r=64, α=128, 3 epochs, lr 2e-5),
`verify_memorization.py` and `report_memorization.py` (gate),
`train_policy.py` and `evaluate_policy_phase2.py` (MM-LEAK attacker).

### 2. Defense ladder and attacks (RQ1–RQ2)

| Experiment | Launcher | Main script |
|---|---|---|
| Defense decomposition, Qwen / LLaVA | `launch_defense_decomposition.sh`, `launch_defense_decomposition_llava.sh` | `defense_decomposition_v1.py`, `defense_decomposition_llava_v1.py` |
| Fixed baselines Direct/DAN/Paraphrase on the ladder (E5) and in-context control (E6) | `launch_ladder_e5_e6.sh` | `matched_baselines_ladder_v1.py`, `incontext_spacing_control_v1.py` |
| Fixed baselines on LLaVA (E5e) | `launch_matched_baselines_llava_v3.sh` | `matched_baselines_llava_v3.py` |
| Sampled decoding (T=0.7, top-p 0.9) | `launch_sampled_baselines_qwen.sh` | `sampled_baselines_qwen_v1.py` |
| MM-LEAK policy at levels S and N (E5c) | `launch_system_only_policy_eval.sh`, `launch_neutral_policy_eval.sh` | `evaluate_policy_system_only_v1.py`, `evaluate_policy_neutral_v1.py` |
| FigStep, 64 and 512 tokens, Qwen / LLaVA | `launch_figstep_ladder.sh`, `launch_figstep_v2.sh`, `launch_figstep_llava_v3.sh` | `figstep_ladder_v1.py`, `figstep_ladder_v2.py`, `figstep_llava_v3.py` |
| PAIR, AutoDAN-Turbo-style, HADES black-box | `launch_adaptive_baselines.sh` | `adaptive_baselines_v1.py` (attacker: Qwen2.5 in `attacker_qwen25_worker.py`) |
| System-prompt wording (E11), Qwen / LLaVA | `launch_system_prompt_sensitivity.sh`, `launch_e11_llava.sh` | `system_prompt_sensitivity_v1.py`, `e11_llava_v3.py` |
| Training without refusals (E16) | `launch_e16.sh` (`TARGET_SEED=101..103`) | `e16_no_refusal.py` |
| Secrets trained under the protected instruction (E17) | `launch_e17.sh` (`TARGET_SEED=101..103`) | `e17_deployer_instruction.py` |

### 3. Why the filter works and where it fails (RQ3)

| Experiment | Launcher | Main script |
|---|---|---|
| Two-turn re-encoding with raw or filtered history (E9) | `launch_twoturn_reencoding.sh` (`ARCH=qwen` or `llava_v3`) | `twoturn_reencoding_v1.py` |
| Natural-language secrets (E10) and their baselines | `launch_e10.sh` (`TARGET_SEED=106`, `107`), `launch_matched_baselines_e10_e14.sh` | `prepare_e10.py`, `e10_measure.py`, `matched_baselines_e10_e14_v1.py` |
| Base64 / reversal / hex re-encoding (E12) | `launch_transform_reencoding.sh` (`PART=incontext` or `twoturn`) | `transform_reencoding_v1.py` |
| Image-bound secrets (E14) | `launch_e14.sh` (`TARGET_SEED=108`, `109`) | `e14_image_key.py` |
| Image variants of the key (E14b) | `launch_e14_variants.sh` | `e14_image_variants.py` |
| Visual attacks on image-bound secrets (E14c) | `launch_e14_visual_attacks.sh` | `e14_visual_attacks.py` |
| Number of training repetitions R=50/10/3 (E15) | `launch_e15.sh` | `e15_repetition.py` |
| Spelled-out re-encoding (E18) | `launch_e18.sh` (`PART=incontext` or `twoturn`) | `e18_spelled_reencoding.py` |

Offline analyses on the saved responses (CPU only, no launcher):

```bash
python scripts/scripts/filter_counterfactual_v1.py      # alternative output filters
python scripts/scripts/filter_false_positives_v1.py     # filter false positives on benign text
python scripts/scripts/body_rescore_v1.py               # rescoring on the private part of the secret
python scripts/scripts/partial_leakage_v1.py            # partial and cross-identity disclosure
python scripts/scripts/paired_ladder_stats.py           # exact McNemar tests with Holm correction
```

### 4. Utility (RQ4)

| Experiment | Launcher | Main script |
|---|---|---|
| Utility and over-refusal of each defense level | `launch_utility_cost.sh` | `utility_cost_v1.py` |
| ScienceQA-IMG and POPE (500 + 500 items) | `launch_utility_benchmark.sh` (`STAGE=fetch` once, then `FAMILY=qwen` / `llava`) | `utility_benchmark_v1.py` |

### 5. Figures and tables

```bash
python TDSC/paper/figures/build_measurement_figures.py
```

It reads the reports in `private/` and writes the figures and the LaTeX
tables (counts only, no secrets or responses).

## Evaluation discipline

- Secrets are derived with HMAC-SHA256 from an evaluator-private key and never
  appear in the public case files that attack code reads.
- Prompts, templates and attacker settings are frozen on the dev targets and
  evaluated once on the eval targets; every target call is counted.
- Each run writes a frozen plan with the SHA-256 of its scripts, data and
  adapters, and resuming stops if any of them changed. Start new runs from a
  clean checkout rather than resuming runs made with an older version.

## Data

- `Dataset/train_integrated_text.jsonl`: Wikipedia and C4 (realnewslike)
  prefix/continuation pairs used as benign replay during fine-tuning and as
  benign text for the filter false-positive test.
- `Dataset/disjoint_lora_seed42/benign_replay_v2/utility.jsonl`: synthetic
  utility tasks (addition, subtraction, multiplication, sorting) with exact
  answers.
- ScienceQA and POPE are downloaded by `launch_utility_benchmark.sh`
  (`derek-thomas/ScienceQA`, `lmms-lab/POPE`).

All secrets are synthetic; no real personal data is used.




