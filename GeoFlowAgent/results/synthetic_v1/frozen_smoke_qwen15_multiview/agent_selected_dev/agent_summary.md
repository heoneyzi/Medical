# GeoFlowAgent closed-loop evaluation (dev)

## Results

| condition | stop_controller | contract_mask | tasks | task_success | goal_reached | correct_stop_rate | contract_valid_action_rate | execution_success_rate | zero_regret_action_rate | mean_regret_label_coverage | assigned_perturbation_tasks | triggered_perturbation_tasks | perturbation_trigger_rate | assigned_task_success_rate | triggered_recovery_rate | mean_regret | mean_planner_calls | mean_total_nfe | mean_latency_seconds | mean_embedding_cache_hit_rate | mean_runtime_embedding_miss_calls | mean_redundant_calls | mean_contract_valid_candidates |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| random_contract_oracle_stop | public_goal_oracle | True | 2 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 8.500000 | 0.000000 | 0.004962 | 0.000000 | 0.000000 | 0.000000 | 1.305556 |
| metric_no_contract_mask_oracle_stop | public_goal_oracle | False | 2 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 8.500000 | 0.000000 | 5.026172 | 0.562500 | 3.500000 | 0.000000 | 1.000000 |
| metric_closed_loop | public_goal_oracle | True | 2 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 8.500000 | 0.000000 | 0.899941 | 0.562500 | 3.500000 | 0.000000 | 1.000000 |
| flow_open_loop | learned | True | 2 | 0.000000 | 0.000000 | 0.000000 | 0.133929 | 0.133929 | 0.133929 | 1.000000 | 0 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.866071 | 1.000000 | 16.000000 | 0.070599 | 1.000000 | 0.000000 | 0.000000 | 2.000000 |
| flow_closed_loop_commit1 | learned | True | 2 | 0.500000 | 0.500000 | 0.500000 | 0.900000 | 0.900000 | 0.900000 | 1.000000 | 0 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.100000 | 7.000000 | 112.000000 | 1.670415 | 0.266667 | 5.000000 | 0.000000 | 1.777778 |
| flow_closed_loop_commit1_no_contract_mask | learned | False | 2 | 0.000000 | 0.000000 | 0.000000 | 0.550000 | 0.550000 | 0.550000 | 1.000000 | 0 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.450000 | 10.000000 | 160.000000 | 2.615673 | 0.200000 | 8.000000 | 0.000000 | 1.850000 |
| flow_closed_loop_commit2 | learned | True | 2 | 0.000000 | 0.000000 | 0.000000 | 0.687500 | 0.687500 | 0.687500 | 1.000000 | 0 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.312500 | 4.000000 | 64.000000 | 0.953211 | 0.250000 | 3.000000 | 0.000000 | 1.875000 |
| flow_compute_matched_blind_replan | learned | True | 2 | 0.000000 | 0.000000 | 0.000000 | 0.155556 | 0.155556 | 0.155556 | 1.000000 | 0 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.844444 | 7.000000 | 112.000000 | 0.422508 | 1.000000 | 0.000000 | 0.000000 | 2.000000 |

## Interpretation

Compare commit-1 with both open-loop and compute-matched blind replanning. Report planner calls, total NFE, and latency with success; otherwise feedback and compute are confounded.
