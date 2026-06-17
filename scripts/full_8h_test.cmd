@echo off
REM ============================================================================
REM full_8h_test.cmd
REM ----------------------------------------------------------------------------
REM End-to-end test of the qwen-en-tutor pipeline. Walks through:
REM   generation -> diversity -> holdout -> train_sft -> eval_intermediate
REM   -> on_policy + DPO -> eval_final -> compare -> deploy test -> GGUF -> diversity
REM
REM Time budget: ~7-8 hours on RTX 3060 + Qwen3.5 4B Q4 LAN teacher/judge.
REM
REM PREREQUISITES (do these once before running):
REM   1) llama-server is up on the LAN with the 4B GGUF and reachable at
REM      config/generation.yaml -> teacher.openai.base_url
REM   2) vendor/models/Qwen_3.5_0.8B (or 4B) contains the HF student weights
REM      and config/training.yaml -> base_model.model_id points at it
REM   3) Optional: edit config/generation.yaml -> generation.n_per_level
REM      (default for this script: 5 seeds x 6 levels = 30)
REM
REM USAGE (cmd.exe, run from the project root):
REM   .venv\Scripts\activate.bat
REM   scripts\full_8h_test.cmd
REM
REM OR copy any individual phase command below and run it manually.
REM Each phase writes a timestamped log to data\_logs\.
REM ============================================================================

setlocal EnableExtensions EnableDelayedExpansion

REM ---- offline-by-default env (mirrors what the entry scripts set) ----------
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
set QWEN_TUTOR_PROMPTS=compact

REM ---- recreate data/ scaffolding if missing --------------------------------
for %%D in (_logs _caches seeds sft_raw sft_filtered dpo_raw dpo_filtered eval_raw eval_filtered holdout eval_results) do (
    if not exist data\%%D mkdir data\%%D
)

set RUN_ID=full8h
echo [%DATE% %TIME%] RUN_ID=%RUN_ID%

REM ===========================================================================
echo [%DATE% %TIME%] === Phase 1/10: generation pipeline (3-4 hours) ===
REM Runs all generation stages: seeds, sft, redirects, register, eval, filter_*
REM ===========================================================================
python scripts\run_generation.py > data\_logs\phase1_generation.log 2>&1
if errorlevel 1 ( echo Phase 1 FAILED & goto :fail )

REM ===========================================================================
echo [%DATE% %TIME%] === Phase 2/10: diversity baseline (5 min) ===
REM Per-pool distribution check: top names, places, categories
REM ===========================================================================
python scripts\run_diversity.py --input-dir data\sft_filtered  > data\_logs\phase2_diversity_sft.log  2>&1
python scripts\run_diversity.py --input-dir data\dpo_filtered  > data\_logs\phase2_diversity_dpo.log  2>&1
python scripts\run_diversity.py --input-dir data\eval_filtered > data\_logs\phase2_diversity_eval.log 2>&1

REM ===========================================================================
echo [%DATE% %TIME%] === Phase 3/10: holdout sampling (30 sec) ===
REM Pulls 3 seeds per level out of data\seeds\ into data\holdout\.
REM Crucially: NOT using --keep-in-seeds, so train pool is genuinely disjoint.
REM ===========================================================================
python scripts\sample_holdout.py --n-per-level 3 --levels A1,A2,B1,B2,C1,C2 > data\_logs\phase3_holdout.log 2>&1
if errorlevel 1 ( echo Phase 3 FAILED & goto :fail )

REM ===========================================================================
echo [%DATE% %TIME%] === Phase 4/10: train SFT (30-60 min) ===
REM Full epoch on the freshly generated training pool.
REM Output: outputs\sft\
REM ===========================================================================
if exist outputs\sft rmdir /S /Q outputs\sft
python scripts\run_training.py --stages train_sft --run-id %RUN_ID% > data\_logs\phase4_train_sft.log 2>&1
if errorlevel 1 ( echo Phase 4 FAILED & goto :fail )

REM ===========================================================================
echo [%DATE% %TIME%] === Phase 5/10: eval_intermediate (60-90 min) ===
REM Holdout evaluation of the SFT-only model. With max_new_tokens_evaluation=4096.
REM Output: data\eval_results\%RUN_ID%_intermediate\
REM ===========================================================================
python scripts\run_training.py --stages eval_intermediate --run-id %RUN_ID% > data\_logs\phase5_eval_intermediate.log 2>&1
if errorlevel 1 ( echo Phase 5 FAILED & goto :fail )

REM ===========================================================================
echo [%DATE% %TIME%] === Phase 6/10: on_policy + filter_dpo + train_dpo (60 min) ===
REM On-policy pair generation against SFT adapter, refilter, then DPO training.
REM Output: outputs\dpo\
REM ===========================================================================
if exist outputs\dpo rmdir /S /Q outputs\dpo
python scripts\run_training.py --stages on_policy_gen,filter_dpo,train_dpo --run-id %RUN_ID% > data\_logs\phase6_on_policy_dpo.log 2>&1
if errorlevel 1 ( echo Phase 6 FAILED & goto :fail )

REM ===========================================================================
echo [%DATE% %TIME%] === Phase 7/10: eval_final + compare (60-90 min) ===
REM Holdout evaluation of the SFT+DPO model, then compare report.
REM Output: data\eval_results\%RUN_ID%_final\compare_report.md
REM ===========================================================================
python scripts\run_training.py --stages eval_final,compare --run-id %RUN_ID% > data\_logs\phase7_eval_final_compare.log 2>&1
if errorlevel 1 ( echo Phase 7 FAILED & goto :fail )

REM ===========================================================================
echo [%DATE% %TIME%] === Phase 8/10: deploy smoke test on SFT and DPO (10 min) ===
REM Walks a scripted conversation through TutorRuntime + runs evaluate_async().
REM Defaults to outputs\dpo; change --adapter inside the script if you want SFT.
REM ===========================================================================
python scripts\test_deploy.py > data\_logs\phase8_test_deploy_dpo.log 2>&1

REM ===========================================================================
echo [%DATE% %TIME%] === Phase 9/10: GGUF export of outputs\dpo (5 min) ===
REM Merges DPO adapter into base, converts to GGUF bf16, quantizes to Q4_K_M.
REM Output: outputs\gguf\qwen-en-tutor-Q4_K_M.gguf
REM ===========================================================================
python scripts\run_gguf_export.py --adapter outputs\dpo > data\_logs\phase9_gguf_export.log 2>&1
if errorlevel 1 ( echo Phase 9 FAILED & goto :fail )

REM ===========================================================================
echo [%DATE% %TIME%] === Phase 10/10: final diversity + summary (5 min) ===
REM Post-pipeline distribution check. Eval pool now carries category metadata.
REM ===========================================================================
python scripts\run_diversity.py --input-dir data\eval_filtered --json-out data\diversity_final_eval.json > data\_logs\phase10_diversity_eval.log 2>&1

echo.
echo [%DATE% %TIME%] === full_8h_test.cmd COMPLETE ===
echo Compare report:  data\eval_results\%RUN_ID%_final\compare_report.md
echo GGUF artifact:   outputs\gguf\qwen-en-tutor-Q4_K_M.gguf
echo Per-phase logs:  data\_logs\phase*.log
exit /b 0

:fail
echo [%DATE% %TIME%] aborted; check the latest data\_logs\phase*.log
exit /b 1
