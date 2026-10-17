SHELL := bash
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := help

PY := uv run python
GPU_START_C ?= 75

MAIN_CKPT ?= checkpoints/gpt2_ppb16/loss_0.40
PLAN_CKPT ?= checkpoints/plan_rep/best
FILL_CKPT ?= checkpoints/fill_rep_long/loss_0.40
BONUS_CKPT ?= checkpoints/abl_melody/step_7000
BONUS_TOKENS ?= data/tokens/bonus_ppb16

TEMP ?= 1.0
TOP_P ?= 1.0
N ?= 20
BARS ?= 32
TASK_B_CONFIGS ?= 0.9,0,1.0 1.0,0,0.95 1.0,0,1.0
PLAN_RULES ?=
PROMPT_DIR ?= data/POP909/prompts/basic-b
WAV ?= --wav
SWEEP_LEVELS ?= loss_1.00 loss_0.90 loss_0.80 loss_0.70 loss_0.60 loss_0.50 loss_0.40 loss_0.30 loss_0.20 best last
SWEEP_PLANS ?= outputs/cand_a/pf_oldplan_fill040/plans.json
SWEEP_FILL_ARGS ?= --n 20 --return_as var --repeat_bias 8 --min_piano_bin 1 --tempo_range 60,125 --cadence

TASK_A_DIR ?= outputs/cand_a/single_t1_p1
PLAN_FILL_DIR ?= outputs/plan_fill
TASK_B_DIR ?= outputs/task_b
PLAN_TASK_B_DIR ?= outputs/task_b_plan
BONUS_DIR ?= outputs/bonus_final

SILENCE_DIRS ?= outputs/exp2/step_6000 outputs/exp2/loss_0.40 outputs/exp2/last outputs/plan_fill/melody_first outputs/plan_fill/piano_first \
	$(wildcard outputs/plan_fill_v2/* outputs/plan_fill_rep/* outputs/task_a_plan*/*)
RETURN_DIRS ?= outputs/exp2/loss_0.40 outputs/exp2/last outputs/plan_fill/melody_first $(wildcard outputs/plan_fill_v2/* outputs/plan_fill_rep/* outputs/task_a_plan*/*)
ADHERENCE_DIRS ?= outputs/plan_fill/melody_first outputs/plan_fill/piano_first $(wildcard outputs/plan_fill_v2/* outputs/plan_fill_rep/* outputs/task_a_plan*/*)

GPU_LOG_QUERY := timestamp,temperature.gpu,utilization.gpu,clocks.sm,power.draw,clocks_throttle_reasons.active

# $(1) run name, $(2) command; logs GPU every 30 s next to the run log
define gpu_run
	@mkdir -p runs
	nvidia-smi --query-gpu=$(GPU_LOG_QUERY) --format=csv -l 30 > runs/$(1)_gpu.csv & logger=$$!; status=0; \
	$(2) > runs/$(1).log 2>&1 || status=$$?; kill $$logger; exit $$status
endef

.PHONY: help setup tokens check stats gpu-cool health \
	train-main train-bonus train-plan train-fill train-plan-rep train-fill-rep \
	task-a task-a-plan task-b task-b-plan bonus \
	eval-task-a eval-task-a-plan eval-task-b eval-task-b-plan eval-bonus analyze \
	grade-fit grade-check grade-predict grade-sweep-fill grade-sweep-plan

help: ## list targets; override settings like make task-a TEMP=1.2 TOP_P=0.95
	@grep -hE '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | sed -E 's/^([a-z-]+):.*## (.*)/  \1|\2/' | column -t -s '|'

setup: ## create the venv from uv.lock
	uv sync

tokens: ## TA tokens, bonus tokens, plan and fill tokens
	$(PY) scripts/ta/tokenization.py --dataset_path data/POP909/training --tok2idx_output data/tokens/ppb16/tok2idx.json --sequences_output data/tokens/ppb16/sequences.json --pos_per_beat 16
	$(PY) scripts/tokenize_bonus.py
	$(PY) scripts/tokenize_plan.py
	$(PY) scripts/tokenize_plan.py --repeat

check: ## decoder round trip and metric baselines
	$(PY) scripts/check_roundtrip.py
	$(PY) scripts/baselines.py

stats: ## dataset statistics, figures, key changes
	$(PY) scripts/data_stats.py
	$(PY) scripts/plot_data_stats.py
	$(PY) scripts/key_changes.py --penalty 4

gpu-cool: ## wait until the GPU is at or below GPU_START_C
	@until t=$$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | head -1); [ "$$t" -le $(GPU_START_C) ]; do \
		echo "GPU $$t C, waiting for <= $(GPU_START_C) C"; sleep 30; done; echo "GPU $$t C, ok"

grade-fit: ## listening-grade predictor: refit on every graded session (run grade-check SESSION=... first for a new one)
	$(PY) scripts/grade_model.py fit

grade-check: ## score a newly graded SESSION with the predictor fit before it
	$(PY) scripts/grade_model.py check --session $(SESSION)

grade-predict: ## predicted grades for every MIDI in DIRS, written to results/grade_pred_$(NAME).csv
	$(PY) scripts/grade_model.py predict $(DIRS) --out results/grade_pred_$(NAME).csv

grade-sweep-fill: gpu-cool ## fill checkpoints of RUN on the same 20 plans, then predicted grade vs loss
	for c in $(SWEEP_LEVELS); do [ -d checkpoints/$(RUN)/$$c ] || continue; \
		$(PY) scripts/generate_plan_fill.py --plan_ckpt $(PLAN_CKPT) --fill_ckpt checkpoints/$(RUN)/$$c --plans_from $(SWEEP_PLANS) \
			$(SWEEP_FILL_ARGS) --output_dir outputs/sweep/$(RUN)/$$c; done
	$(PY) scripts/grade_model.py predict outputs/sweep/$(RUN)/* --out results/grade_pred_$(RUN).csv --plot results/figures/grade_vs_loss_$(RUN).png

grade-sweep-plan: gpu-cool ## plan checkpoints of RUN with one fill checkpoint, then predicted grade vs loss
	for c in $(SWEEP_LEVELS); do [ -d checkpoints/$(RUN)/$$c ] || continue; \
		$(PY) scripts/generate_plan_fill.py --plan_ckpt checkpoints/$(RUN)/$$c --fill_ckpt $(FILL_CKPT) --form I4A8B8A8E4 \
			$(SWEEP_FILL_ARGS) --output_dir outputs/sweep/$(RUN)/$$c; done
	$(PY) scripts/grade_model.py predict outputs/sweep/$(RUN)/* --ckpt_key plan_ckpt --out results/grade_pred_$(RUN).csv --plot results/figures/grade_vs_loss_$(RUN).png

health: ## GPU, running jobs, stalled logs, disk
	bash scripts/health_check.sh || true

train-main: gpu-cool ## main model for TASK A and TASK B
	$(call gpu_run,gpt2_ppb16,$(PY) scripts/train.py --run_name gpt2_ppb16)

train-bonus: gpu-cool ## melody-only bonus model (headline BONUS TASK run)
	$(call gpu_run,abl_melody,$(PY) scripts/train.py --run_name abl_melody --tokens_dir $(BONUS_TOKENS) --max_steps 12000 --keep_every 1000 --save_at 0.0)

train-plan: gpu-cool ## plan model of the plan-fill pipeline
	$(call gpu_run,plan_v1,$(PY) scripts/train.py --run_name plan_v1 --tokens_dir data/tokens/plan --ctx 1024 --n_layer 6 --n_head 6 --n_embd 384 --dropout 0.2 --batch_size 16 --lr 5e-4 --warmup 200 --max_steps 4000 --max_shift 0 --val_every 100 --save_every 1000 --save_best --save_at 0.01)

train-fill: gpu-cool ## fill model of the plan-fill pipeline
	$(call gpu_run,fill_v1,$(PY) scripts/train.py --run_name fill_v1 --tokens_dir data/tokens/fill --fill --max_shift 0 --save_best --keep_every 2000 --max_steps 12000 --val_every 500)

train-plan-rep: gpu-cool ## plan model with melody repeat labels (Rep_*)
	$(call gpu_run,plan_rep,$(PY) scripts/train.py --run_name plan_rep --tokens_dir data/tokens/plan_rep --ctx 1024 --n_layer 6 --n_head 6 --n_embd 384 --dropout 0.2 --batch_size 16 --lr 5e-4 --warmup 200 --max_steps 4000 --max_shift 0 --val_every 100 --save_every 1000 --save_best --save_at 0.01)

train-fill-rep: gpu-cool ## fill model with melody repeat labels (Rep_*)
	$(call gpu_run,fill_rep,$(PY) scripts/train.py --run_name fill_rep --tokens_dir data/tokens/fill_rep --fill --max_shift 0 --save_best --keep_every 2000 --max_steps 10000 --val_every 500)

task-a: gpu-cool ## TASK A with the main model
	$(PY) scripts/generate_uncond.py --ckpt $(MAIN_CKPT) --output_dir $(TASK_A_DIR) --n $(N) --bars $(BARS) --temperature $(TEMP) --top_p $(TOP_P) $(WAV)

task-a-plan: gpu-cool ## TASK A with plan-fill, melody-first then piano-first on the same plans
	$(PY) scripts/generate_plan_fill.py --plan_ckpt $(PLAN_CKPT) --fill_ckpt $(FILL_CKPT) --output_dir $(PLAN_FILL_DIR)/melody_first --order melody_first --n $(N) --bars $(BARS) --temperature $(TEMP) --top_p $(TOP_P) $(PLAN_RULES) $(WAV)
	$(PY) scripts/generate_plan_fill.py --plan_ckpt $(PLAN_CKPT) --fill_ckpt $(FILL_CKPT) --output_dir $(PLAN_FILL_DIR)/piano_first --order piano_first --plans_from $(PLAN_FILL_DIR)/melody_first/plans.json --n $(N) --bars $(BARS) --temperature $(TEMP) --top_p $(TOP_P) $(PLAN_RULES) $(WAV)

task-b: gpu-cool ## TASK B continuations of the 3 prompts, one folder per config
	$(PY) scripts/generate_cont.py --ckpt $(MAIN_CKPT) --output_dir $(TASK_B_DIR) $(foreach c,$(TASK_B_CONFIGS),--config $(c)) $(WAV)

task-b-plan: gpu-cool ## TASK B with plan-fill: one plan per prompt, filled once per config (melody enters by bar 9)
	$(PY) scripts/generate_plan_fill.py --plan_ckpt $(PLAN_CKPT) --fill_ckpt $(FILL_CKPT) --prompt_dir $(PROMPT_DIR) --output_dir $(PLAN_TASK_B_DIR) --melody_by_bar 9 $(foreach c,$(TASK_B_CONFIGS),--config $(c)) $(PLAN_RULES) $(WAV)

bonus: gpu-cool ## BONUS TASK accompaniment for the 10 prompts, then the TA eval
	$(PY) scripts/generate_bonus.py --ckpt $(BONUS_CKPT) --tok2idx $(BONUS_TOKENS)/tok2idx.json --output_dir $(BONUS_DIR) --top_p 0.95 $(WAV)
	$(MAKE) eval-bonus

eval-task-a: ## H1/H4/GS, NLL, copy rate for TASK A
	$(PY) scripts/eval_generated.py $(TASK_A_DIR) --summary_csv results/task_a.csv

eval-task-a-plan: ## same metrics for both plan-fill orders
	$(PY) scripts/eval_generated.py $(PLAN_FILL_DIR)/melody_first $(PLAN_FILL_DIR)/piano_first --summary_csv results/plan_fill.csv

eval-task-b: ## metrics and prompt fade for TASK B (copy rate skips the 8 prompt bars)
	$(PY) scripts/eval_generated.py $(TASK_B_DIR)/* --skip_bars 8 --summary_csv results/task_b.csv
	$(PY) scripts/prompt_fade.py $(TASK_B_DIR)/* --out results/task_b_fade.csv

eval-task-b-plan: ## same metrics for TASK B with plan-fill
	$(PY) scripts/eval_generated.py $(PLAN_TASK_B_DIR)/* --skip_bars 8 --summary_csv results/task_b_plan.csv
	$(PY) scripts/prompt_fade.py $(PLAN_TASK_B_DIR)/* --out results/task_b_plan_fade.csv

analyze: ## silence, theme return and plan adherence tables for the folders in SILENCE_DIRS, RETURN_DIRS, ADHERENCE_DIRS
	$(PY) scripts/silence_stats.py $(SILENCE_DIRS)
	$(PY) scripts/return_stats.py $(RETURN_DIRS)
	$(PY) scripts/adherence.py $(ADHERENCE_DIRS)

eval-bonus: ## TA eval_bonus.py (runs from scripts/ta, which it needs)
	cd scripts/ta && $(PY) eval_bonus.py --prompt_dir ../../data/POP909/prompts/bonus --gen_midi_dir ../../$(BONUS_DIR) --output_csv ../../$(BONUS_DIR)/metrics.csv
