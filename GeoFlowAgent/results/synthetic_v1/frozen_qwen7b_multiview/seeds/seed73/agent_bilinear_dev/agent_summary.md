# GeoFlowAgent closed-loop evaluation (dev)

## Results

| condition | stop_controller | contract_mask | tasks | task_success | goal_reached | correct_stop_rate | contract_valid_action_rate | execution_success_rate | zero_regret_action_rate | mean_regret_label_coverage | assigned_perturbation_tasks | triggered_perturbation_tasks | perturbation_trigger_rate | assigned_task_success_rate | triggered_recovery_rate | mean_regret | mean_planner_calls | mean_total_nfe | mean_latency_seconds | mean_embedding_cache_hit_rate | mean_runtime_embedding_miss_calls | mean_redundant_calls | mean_contract_valid_candidates |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| random_contract_oracle_stop | public_goal_oracle | True | 2 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 8.500000 | 0.000000 | 0.004379 | 0.000000 | 0.000000 | 0.000000 | 1.118056 |
| metric_no_contract_mask_oracle_stop | public_goal_oracle | False | 2 | 0.500000 | 0.500000 | 0.500000 | 0.562500 | 0.562500 | 0.562500 | 1.000000 | 0 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.437500 | 12.000000 | 0.000000 | 10.884898 | 0.531250 | 7.500000 | 0.000000 | 1.031250 |
| metric_closed_loop | public_goal_oracle | True | 2 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 8.500000 | 0.000000 | 1.447167 | 0.562500 | 3.500000 | 0.000000 | 1.000000 |
| flow_open_loop | learned | True | 2 | 0.000000 | 0.000000 | 0.000000 | 0.062500 | 0.062500 | 0.062500 | 1.000000 | 0 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.937500 | 1.000000 | 16.000000 | 0.069413 | 1.000000 | 0.000000 | 0.000000 | 1.562500 |
| flow_closed_loop_commit1 | learned | True | 2 | 0.500000 | 0.500000 | 0.500000 | 0.500000 | 0.500000 | 0.500000 | 1.000000 | 0 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.500000 | 4.500000 | 72.000000 | 0.260937 | 1.000000 | 0.000000 | 0.000000 | 1.500000 |
| flow_closed_loop_commit1_no_contract_mask | learned | False | 2 | 0.000000 | 0.000000 | 0.000000 | 0.050000 | 0.050000 | 0.050000 | 1.000000 | 0 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.950000 | 5.500000 | 88.000000 | 1.886039 | 0.600000 | 4.000000 | 0.000000 | 1.550000 |
| flow_closed_loop_commit2 | learned | True | 2 | 0.000000 | 0.000000 | 0.000000 | 0.277778 | 0.277778 | 0.277778 | 1.000000 | 0 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.722222 | 3.000000 | 48.000000 | 0.985279 | 0.600000 | 2.000000 | 0.000000 | 1.555556 |
| flow_compute_matched_blind_replan | learned | True | 2 | 0.000000 | 0.000000 | 0.000000 | 0.142857 | 0.142857 | 0.142857 | 1.000000 | 0 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.857143 | 4.500000 | 72.000000 | 0.260511 | 1.000000 | 0.000000 | 0.000000 | 1.571429 |

## Interpretation

Compare commit-1 with both open-loop and compute-matched blind replanning. Report planner calls, total NFE, and latency with success; otherwise feedback and compute are confounded.
