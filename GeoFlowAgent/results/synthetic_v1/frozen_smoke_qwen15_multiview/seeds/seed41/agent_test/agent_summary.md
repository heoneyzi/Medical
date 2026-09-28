# GeoFlowAgent closed-loop evaluation (test)

## Results

| condition | stop_controller | contract_mask | tasks | task_success | goal_reached | correct_stop_rate | contract_valid_action_rate | execution_success_rate | zero_regret_action_rate | mean_regret_label_coverage | assigned_perturbation_tasks | triggered_perturbation_tasks | perturbation_trigger_rate | assigned_task_success_rate | triggered_recovery_rate | mean_regret | mean_planner_calls | mean_total_nfe | mean_latency_seconds | mean_embedding_cache_hit_rate | mean_runtime_embedding_miss_calls | mean_redundant_calls | mean_contract_valid_candidates |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| random_contract_oracle_stop | public_goal_oracle | True | 3 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 8.666667 | 0.000000 | 0.004849 | 0.000000 | 0.000000 | 0.000000 | 1.259259 |
| metric_no_contract_mask_oracle_stop | public_goal_oracle | False | 3 | 0.000000 | 0.000000 | 0.000000 | 0.208333 | 0.208333 | 0.208333 | 1.000000 | 0 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.791667 | 16.000000 | 0.000000 | 6.824532 | 0.083333 | 14.666667 | 0.000000 | 1.395833 |
| metric_closed_loop | public_goal_oracle | True | 3 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 8.666667 | 0.000000 | 1.594680 | 0.172619 | 6.333333 | 0.000000 | 1.222222 |
| flow_open_loop | learned | True | 3 | 0.000000 | 0.000000 | 0.000000 | 0.239286 | 0.239286 | 0.239286 | 1.000000 | 0 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.760714 | 1.000000 | 16.000000 | 0.068339 | 1.000000 | 0.000000 | 0.000000 | 1.733333 |
| flow_closed_loop_commit1 | learned | True | 3 | 0.333333 | 0.333333 | 0.333333 | 0.904762 | 0.904762 | 0.904762 | 1.000000 | 0 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.095238 | 7.333333 | 117.333333 | 1.410407 | 0.464286 | 4.000000 | 0.000000 | 1.232143 |
| flow_closed_loop_commit1_no_contract_mask | learned | False | 3 | 0.000000 | 0.000000 | 0.000000 | 0.125000 | 0.125000 | 0.125000 | 1.000000 | 0 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.875000 | 16.000000 | 256.000000 | 4.862282 | 0.104167 | 14.333333 | 0.000000 | 2.000000 |
| flow_closed_loop_commit2 | learned | True | 3 | 0.333333 | 0.333333 | 0.333333 | 0.665705 | 0.665705 | 0.665705 | 1.000000 | 0 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.334295 | 5.000000 | 80.000000 | 1.303832 | 0.205556 | 4.000000 | 0.000000 | 1.327561 |
| flow_compute_matched_blind_replan | learned | True | 3 | 0.000000 | 0.000000 | 0.000000 | 0.178571 | 0.178571 | 0.178571 | 1.000000 | 0 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.821429 | 7.333333 | 117.333333 | 0.438908 | 1.000000 | 0.000000 | 0.000000 | 1.916667 |

## Interpretation

Compare commit-1 with both open-loop and compute-matched blind replanning. Report planner calls, total NFE, and latency with success; otherwise feedback and compute are confounded.
