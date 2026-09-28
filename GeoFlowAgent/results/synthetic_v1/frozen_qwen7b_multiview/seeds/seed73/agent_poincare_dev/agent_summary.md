# GeoFlowAgent closed-loop evaluation (dev)

## Results

| condition | stop_controller | contract_mask | tasks | task_success | goal_reached | correct_stop_rate | contract_valid_action_rate | execution_success_rate | zero_regret_action_rate | mean_regret_label_coverage | assigned_perturbation_tasks | triggered_perturbation_tasks | perturbation_trigger_rate | assigned_task_success_rate | triggered_recovery_rate | mean_regret | mean_planner_calls | mean_total_nfe | mean_latency_seconds | mean_embedding_cache_hit_rate | mean_runtime_embedding_miss_calls | mean_redundant_calls | mean_contract_valid_candidates |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| random_contract_oracle_stop | public_goal_oracle | True | 2 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 8.500000 | 0.000000 | 0.004363 | 0.000000 | 0.000000 | 0.000000 | 1.118056 |
| metric_no_contract_mask_oracle_stop | public_goal_oracle | False | 2 | 0.500000 | 0.500000 | 0.500000 | 0.531250 | 0.531250 | 0.531250 | 1.000000 | 0 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.468750 | 12.000000 | 0.000000 | 10.692488 | 0.531250 | 7.500000 | 0.000000 | 1.031250 |
| metric_closed_loop | public_goal_oracle | True | 2 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 8.500000 | 0.000000 | 1.471330 | 0.562500 | 3.500000 | 0.000000 | 1.000000 |
| flow_open_loop | learned | True | 2 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 1.000000 | 16.000000 | 0.065261 | 1.000000 | 0.000000 | 0.000000 | 2.000000 |
| flow_closed_loop_commit1 | learned | True | 2 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 1.000000 | 16.000000 | 0.061501 | 1.000000 | 0.000000 | 0.000000 | 2.000000 |
| flow_closed_loop_commit1_no_contract_mask | learned | False | 2 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 1.000000 | 16.000000 | 0.060129 | 1.000000 | 0.000000 | 0.000000 | 2.000000 |
| flow_closed_loop_commit2 | learned | True | 2 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 1.000000 | 16.000000 | 0.060223 | 1.000000 | 0.000000 | 0.000000 | 2.000000 |
| flow_compute_matched_blind_replan | learned | True | 2 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 1.000000 | 16.000000 | 0.060144 | 1.000000 | 0.000000 | 0.000000 | 2.000000 |

## Interpretation

Compare commit-1 with both open-loop and compute-matched blind replanning. Report planner calls, total NFE, and latency with success; otherwise feedback and compute are confounded.
