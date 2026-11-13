# RLHF on Reddit TL;DR — one command per task. Run from the repo root.
PY ?= python
HF_USER ?= latent-hassan

.PHONY: setup starter task1 sft reward ppo dpo eval human-sheet human-score samples analysis tables figures app smoke upload download clean

setup:        ## install dependencies
	pip install -r requirements.txt
starter:      ## Task 1.1 — run the unchanged starter check
	cd starter && $(PY) check_starter.py
task1:        ## Task 1.2–1.3 — data statistics + summary-mask check
	$(PY) common.py
sft:          ## Task 2
	$(PY) train_sft.py
reward:       ## Task 3
	$(PY) train_reward.py
ppo:          ## Task 4 (RESUME=1 make ppo to continue after a disconnect)
	$(PY) train_ppo.py
dpo:          ## Task 5
	$(PY) train_dpo.py
eval:         ## Task 6
	$(PY) evaluate.py run
human-sheet:  ## §3.2 — blind sheets, one per rater
	$(PY) evaluate.py human-sheet --raters 2
human-score:  ## §3.2 — after both sheets are filled in
	$(PY) evaluate.py human-score --labels "Rater 1 (author)" "Rater 2 (external, blind)" --note "$(NOTE)"
samples:      ## §4.6 — candidates (or IDS="3 17 ...")
	$(PY) evaluate.py samples $(if $(IDS),--ids $(IDS),)
analysis:     ## Task 7 numbers → results/analysis_data.md
	$(PY) evaluate.py analysis-data
tables:
	$(PY) evaluate.py tables
figures:
	$(PY) evaluate.py figures
app:          ## Task 8.1
	$(PY) app/app.py

smoke:        ## 2-minute end-to-end dry run on tiny slices (writes to checkpoints/ and results/ — run `make clean` after)
	SMOKE=1 $(PY) train_sft.py && SMOKE=1 $(PY) train_reward.py && SMOKE=1 $(PY) train_dpo.py && SMOKE=1 $(PY) train_ppo.py && SMOKE=1 $(PY) evaluate.py run

upload:       ## push checkpoints to the Hugging Face Hub (huggingface-cli login first)
	for s in sft reward ppo dpo; do huggingface-cli upload $(HF_USER)/smollm2-135m-tldr-$$s checkpoints/$$s . ; done
download:     ## fetch released checkpoints into checkpoints/
	for s in sft reward ppo dpo; do huggingface-cli download $(HF_USER)/smollm2-135m-tldr-$$s --local-dir checkpoints/$$s ; done

clean:        ## remove run artefacts (keeps the data cache)
	rm -rf checkpoints results/logs/*.json* results/figures/fig[1-4]*.png results/correctness_checks.txt results/tables.md \
	       results/eval_summaries.jsonl results/sft_samples.md results/samples.md results/analysis_data.md \
	       results/human_eval_blind_r*.csv results/human_eval_key.json results/human_eval.csv
