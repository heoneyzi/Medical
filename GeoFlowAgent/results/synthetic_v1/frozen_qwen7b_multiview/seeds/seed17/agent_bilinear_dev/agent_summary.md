# GeoFlowAgent closed-loop evaluation (dev)

## Results

| condition | stop_controller | contract_mask | tasks | task_success | goal_reached | correct_stop_rate | contract_valid_action_rate | execution_success_rate | zero_regret_action_rate | mean_regret_label_coverage | assigned_perturbation_tasks | triggered_perturbation_tasks | perturbation_trigger_rate | assigned_task_success_rate | triggered_recovery_rate | mean_regret | mean_planner_calls | mean_total_nfe | mean_latency_seconds | mean_embedding_cache_hit_rate | mean_runtime_embedding_miss_calls | mean_redundant_calls | mean_contract_valid_candidates |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| random_contract_oracle_stop | public_goal_oracle | True | 2 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 8.500000 | 0.000000 | 0.004931 | 0.000000 | 0.000000 | 0.000000 | 1.305556 |
| metric_no_contract_mask_oracle_stop | public_goal_oracle | False | 2 | 0.500000 | 0.500000 | 0.500000 | 0.562500 | 0.562500 | 0.562500 | 1.000000 | 0 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.437500 | 12.000000 | 0.000000 | 11.702937 | 0.102679 | 10.500000 | 0.000000 | 1.093750 |
| metric_closed_loop | public_goal_oracle | True | 2 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 8.500000 | 0.000000 | 2.567267 | 0.133929 | 6.500000 | 0.000000 | 1.062500 |
| flow_open_loop | learned | True | 2 | 0.000000 | 0.000000 | 0.000000 | 0.166667 | 0.166667 | 0.166667 | 1.000000 | 0 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.833333 | 1.000000 | 16.000000 | 0.069474 | 1.000000 | 0.000000 | 0.000000 | 1.166667 |
| flow_closed_loop_commit1 | learned | True | 2 | 0.500000 | 0.500000 | 0.500000 | 0.937500 | 0.937500 | 0.937500 | 1.000000 | 0 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.062500 | 8.000000 | 128.000000 | 1.893342 | 0.562500 | 3.500000 | 0.000000 | 1.062500 |
| flow_closed_loop_commit1_no_contract_mask | learned | False | 2 | 0.000000 | 0.000000 | 0.000000 | 0.062500 | 0.062500 | 0.062500 | 1.000000 | 0 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.937500 | 16.000000 | 256.000000 | 7.968359 | 0.093750 | 14.500000 | 0.000000 | 1.062500 |
| flow_closed_loop_commit2 | learned | True | 2 | 0.500000 | 0.500000 | 0.500000 | 0.600000 | 0.600000 | 0.600000 | 1.000000 | 0 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.400000 | 6.500000 | 104.000000 | 3.146007 | 0.162500 | 5.500000 | 0.000000 | 1.050000 |
| flow_compute_matched_blind_replan | learned | True | 2 | 0.000000 | 0.000000 | 0.000000 | 0.250000 | 0.250000 | 0.250000 | 1.000000 | 0 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.750000 | 8.000000 | 128.000000 | 0.467791 | 1.000000 | 0.000000 | 0.000000 | 1.125000 |

## Interpretation

Compare commit-1 with both open-loop and compute-matched blind replanning. Report planner calls, total NFE, and latency with success; otherwise feedback and compute are confounded.
