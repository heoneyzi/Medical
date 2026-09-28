# GeoFlowAgent closed-loop evaluation (dev)

## Results

| condition | stop_controller | contract_mask | tasks | task_success | goal_reached | correct_stop_rate | contract_valid_action_rate | execution_success_rate | zero_regret_action_rate | mean_regret_label_coverage | assigned_perturbation_tasks | triggered_perturbation_tasks | perturbation_trigger_rate | assigned_task_success_rate | triggered_recovery_rate | mean_regret | mean_planner_calls | mean_total_nfe | mean_latency_seconds | mean_embedding_cache_hit_rate | mean_runtime_embedding_miss_calls | mean_redundant_calls | mean_contract_valid_candidates |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| random_contract_oracle_stop | public_goal_oracle | True | 2 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 8.500000 | 0.000000 | 0.004363 | 0.000000 | 0.000000 | 0.000000 | 1.125000 |
| metric_no_contract_mask_oracle_stop | public_goal_oracle | False | 2 | 1.000000 | 1.000000 | 1.000000 | 0.950000 | 0.950000 | 0.950000 | 1.000000 | 0 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.050000 | 9.000000 | 0.000000 | 9.015750 | 0.555556 | 4.000000 | 0.000000 | 1.000000 |
| metric_closed_loop | public_goal_oracle | True | 2 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 8.500000 | 0.000000 | 1.463734 | 0.562500 | 3.500000 | 0.000000 | 1.000000 |
| flow_open_loop | learned | True | 2 | 0.000000 | 0.000000 | 0.000000 | 0.383333 | 0.383333 | 0.383333 | 1.000000 | 0 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.616667 | 1.000000 | 16.000000 | 0.065221 | 1.000000 | 0.000000 | 0.000000 | 1.283333 |
| flow_closed_loop_commit1 | learned | True | 2 | 0.000000 | 0.000000 | 0.000000 | 0.833333 | 0.833333 | 0.833333 | 1.000000 | 0 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.166667 | 6.000000 | 96.000000 | 1.119499 | 0.666667 | 2.000000 | 0.000000 | 1.250000 |
| flow_closed_loop_commit1_no_contract_mask | learned | False | 2 | 0.000000 | 0.000000 | 0.000000 | 0.104167 | 0.104167 | 0.104167 | 1.000000 | 0 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.895833 | 14.000000 | 224.000000 | 6.575833 | 0.145833 | 12.000000 | 0.000000 | 1.104167 |
| flow_closed_loop_commit2 | learned | True | 2 | 0.000000 | 0.000000 | 0.000000 | 0.625000 | 0.625000 | 0.625000 | 1.000000 | 0 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.375000 | 3.500000 | 56.000000 | 1.116924 | 0.291667 | 2.500000 | 0.000000 | 1.208333 |
| flow_compute_matched_blind_replan | learned | True | 2 | 0.000000 | 0.000000 | 0.000000 | 0.416667 | 0.416667 | 0.416667 | 1.000000 | 0 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.583333 | 6.000000 | 96.000000 | 0.354066 | 1.000000 | 0.000000 | 0.000000 | 1.416667 |

## Interpretation

Compare commit-1 with both open-loop and compute-matched blind replanning. Report planner calls, total NFE, and latency with success; otherwise feedback and compute are confounded.
