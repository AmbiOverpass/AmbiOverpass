# AmbiOverpass

Code, benchmark and result files for the paper

**AmbiOverpass: Execution-Aware Ambiguity Topology and Shapley-Guided Planning for Text-to-OverpassQL Generation**
Haoran Luo, Qi Shao ([ORCID](https://orcid.org/0009-0001-0493-7840)), Chen Jason Zhang, Yuanfeng Song, Raymond Chi-Wing Wong.

Project page and online appendix: https://ambioverpass.github.io/

The repository contains

* **AmbiOverpass-Bench**: 1,754 planning instances in three subsets and 254 rewriting cases (`benchmark/`);
* the **AmbiOverpass** implementation: superset execution, result classes of the ambiguity topology, exact Shapley values and the Shapley look-ahead planner, together with all baselines (`ambitopo/`);
* the scripts that build the benchmark, compute the LLM priors, run the planners and produce every number in Tables III–V (`scripts/`);
* the per-request result files of the paper, the cached LLM priors and their calibration (`out/`), and the cached Overpass execution results (release asset).

Everything runs with Python 3.11 or later and the standard library only. An LLM endpoint is needed only to regenerate the priors and to run the two LLM-based baselines.

## Quick start: rebuild the tables of the paper

```bash
git clone https://github.com/AmbiOverpass/AmbiOverpass.git
cd AmbiOverpass
python3 scripts/make_tables.py
```

The script reads the per-request results in `out/`, prints Table III (single object) and Table IV (single and two landmarks), and writes all numbers of the experimental section, including the paired bootstrap confidence intervals, the ablation and the analysis by number of slots, to `out/paper_numbers.json`. It runs in a few seconds without network access.

## Rerun the planners offline

The planning experiments read candidate results from a cache of Overpass responses, so no database request is needed.

```bash
# download the execution cache (72 MB) from the GitHub release and unpack it into out/
curl -L -o overpass_cache.tar.gz https://github.com/AmbiOverpass/AmbiOverpass/releases/download/v1.0/overpass_cache.tar.gz
tar xzf overpass_cache.tar.gz -C out/
export AMBITOPO_JACCARD=1.0

# two-landmark test split, Table IV (AmbiOverpass = ShapleyRollout, EC2-Greedy = Greedy, EIG = Info gain (Qiu))
SPLIT=test M_BENCH=m4 RAW_FILE=out/llm_prior_raw_landmark_iterC1.json CALIB_FILE=out/prior_calib_boot_iterC1.json \
  OUT_TAG=_rerun POLS="ShapleyRollout;Greedy;Info gain (Qiu)" python3 scripts/eval_m3.py llmcal

# held-out split of the same subset
M_BENCH=m4x AMBI_EXTRA_ROWS=out/onl_rows.jsonl RAW_FILE=out/llm_prior_raw_test2_iterC1.json \
  CALIB_FILE=out/prior_calib_boot_iterC1.json OUT_TAG=_rerun python3 scripts/eval_m3.py llmcal

# single-object test split, Table III, budget K = 2
MT_VARIANT=m2b MT_SPLIT=test MT_PRIOR=llmcal MT_TRUTH=gold MT_PRIOR_FILE=out/llm_prior_m2_iter4.json \
  MT_CALIB_FILE=out/prior_calib_m2_iter4.json OUT_TAG=_rerun python3 scripts/main_table.py 2 5
```

`M_BENCH` selects the subset: `m3` and `m3x` are the test and held-out splits of the single-landmark subset, `m4` and `m4x` those of the two-landmark subset. Omitting `POLS` runs every method. Each run writes per-request results to `out/` in the same format as the files used by `make_tables.py`.

We checked these commands on a clean copy of the repository on a different machine (macOS, Python 3.11) from the one that produced the paper results (Linux, Python 3.12). Per request and seed, EX and the number of questions match the result files of the paper except on three requests, where two actions have equal value up to floating-point rounding and the tie is broken differently: one held-out two-landmark request for AmbiOverpass at K = 2 and 3, and one single-object request for EIG at K = 2. The pooled two-landmark EX of AmbiOverpass becomes 0.377 instead of 0.380 at K = 2 and 0.528 instead of 0.529 at K = 3, and the single-object EX of EIG at K = 2 becomes 0.863 instead of 0.865; all other reported values are unchanged.

The LLM planner and the ReAct agent call Qwen3-14B. To run them, serve the model with an OpenAI-compatible server, for example

```bash
vllm serve Qwen/Qwen3-14B-AWQ --served-model-name qwen3-14b --port 8000
export AMBITOPO_LLM_BASE_URL=http://localhost:8000/v1 AMBITOPO_LLM_MODEL=qwen3-14b \
       AMBITOPO_LLM_KEY_ENV=LOCAL_KEY LOCAL_KEY=EMPTY AMBITOPO_LLM_NOTHINK=1
```

## Benchmark format

All files are JSON Lines, one instance per line.

| File | Instances | Splits | Content |
| --- | --- | --- | --- |
| `benchmark/single_object.jsonl` | 976 | dev 583, test 393 | ambiguous request `x_vague`, precise request `x_clear`, intended query `gold_query`, weakened slots `weakened` |
| `benchmark/single_landmark.jsonl` | 343 | dev 109, test 62, held-out 172 | ambiguous and precise request, landmark kind, intended value per slot `gold`, slot list, number of non-empty interpretations `hyps` and result classes `classes` |
| `benchmark/two_landmark.jsonl` | 435 | dev 153, test 70, held-out 212 | as above with two landmark kinds |
| `benchmark/rewriting.jsonl` | 254 | test | vague request `x_hard`, precise request, intended query and tag, applied perturbations and the rewrite operations needed to recover the tag |

Field `gold` lists the index of the intended value for each slot in `slots`, in the order in which `ambitopo` builds the candidate values. The landmark subsets reference their OverpassNL source request by `sample_id`; the source request–query pairs are in `data/source/` (dev and test splits) and `out/onl_rows.jsonl` (held-out split). The candidate interpretations, their execution results and the result classes of an instance are rebuilt from these files and the execution cache by `ambitopo.data` and `ambitopo.near`, as done at the start of `scripts/eval_m3.py`.

## Repository layout

| Path | Content |
| --- | --- |
| `ambitopo/query.py` | parsing and rendering of OverpassQL queries for any choice of slot values |
| `ambitopo/overpass.py` | Overpass client with disk cache; construction of superset queries |
| `ambitopo/readings.py`, `topology.py`, `near.py` | candidate values per slot, slot dependencies, proximity slots |
| `ambitopo/belief.py` | superset execution, local splitting into interpretations, result classes, clarification states |
| `ambitopo/shapley.py`, `planners.py` | coalition values, exact Shapley values, Shapley look-ahead planner (`ShapleyRollout`) and all baselines |
| `ambitopo/episode.py` | simulated user with no-preference answers and the interaction loop |
| `scripts/build_m2.py`, `build_m3.py`, `build_m4.py`, `make_hard.py`, `qc_hard.py` | construction and quality control of the benchmark subsets |
| `scripts/llm_prior.py`, `landmark_prior.py`, `prior_order_check.py`, `iter_tag_sampling.py`, `iter_tag_m2.py`, `calibrate_prior.py`, `calib_cv.py` | LLM priors averaged over option orders, sampled-choice priors and their calibration on the dev split |
| `scripts/main_table.py`, `eval_m3.py` | planning experiments (Tables III and IV) |
| `scripts/ablation_final.py` | ablation (Table V) |
| `scripts/coverage_check.py`, `coverage_controls.py` | candidate generation on the rewriting subset |
| `scripts/exec_cost.py`, `latency*.py` | execution cost and planning latency |
| `scripts/check_improvement.py`, `theory_checks.py` | numerical checks of the policy-improvement guarantee and the Shapley properties |
| `scripts/make_tables.py` | all numbers of the experimental section from the result files |
| `tests/` | unit tests (`python3 tests/test_core.py`, `python3 tests/test_shapley.py`) |

## Settings of the paper

Question cost λ = 0.05, no-preference rate ε = 0.2, budgets K = 1, 2, 3 and no budget, five seeds per request with a shared user random stream across methods, result classes by identical object sets (`AMBITOPO_JACCARD=1.0`). The prior combines LLM probabilities averaged over four option orders with sampled-choice estimates for the target tag (and for the element type of single-object requests), calibrated on the dev split. These priors are stored in `out/llm_prior_*.json` and `out/prior_calib_*.json`. Execution results were retrieved from a public Overpass server and cached.

## License

The code is released under the MIT License. The benchmark annotations are released under CC BY 4.0. The source requests and reference queries come from OverpassNL (Staniek et al., TACL 2024), and the execution results are OpenStreetMap data, © OpenStreetMap contributors, available under the Open Database License.

## Citation

```bibtex
@inproceedings{luo2027ambioverpass,
  title     = {{AmbiOverpass}: Execution-Aware Ambiguity Topology and Shapley-Guided Planning for Text-to-{OverpassQL} Generation},
  author    = {Luo, Haoran and Shao, Qi and Zhang, Chen Jason and Song, Yuanfeng and Wong, Raymond Chi-Wing},
  year      = {2027}
}
```
