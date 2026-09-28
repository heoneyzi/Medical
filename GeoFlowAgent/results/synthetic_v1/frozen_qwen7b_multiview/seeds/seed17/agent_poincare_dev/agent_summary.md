# GeoFlowAgent closed-loop evaluation (dev)

## Results

| condition | stop_controller | contract_mask | tasks | task_success | goal_reached | correct_stop_rate | contract_valid_action_rate | execution_success_rate | zero_regret_action_rate | mean_regret_label_coverage | assigned_perturbation_tasks | triggered_perturbation_tasks | perturbation_trigger_rate | assigned_task_success_rate | triggered_recovery_rate | mean_regret | mean_planner_calls | mean_total_nfe | mean_latency_seconds | mean_embedding_cache_hit_rate | mean_runtime_embedding_miss_calls | mean_redundant_calls | mean_contract_valid_candidates |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| random_contract_oracle_stop | public_goal_oracle | True | 2 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 8.500000 | 0.000000 | 0.004959 | 0.000000 | 0.000000 | 0.000000 | 1.305556 |
| metric_no_contract_mask_oracle_stop | public_goal_oracle | False | 2 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 8.500000 | 0.000000 | 10.218152 | 0.133929 | 6.500000 | 0.000000 | 1.062500 |
| metric_closed_loop | public_goal_oracle | True | 2 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 8.500000 | 0.000000 | 2.562710 | 0.133929 | 6.500000 | 0.000000 | 1.062500 |
| flow_open_loop | learned | True | 2 | 0.000000 | 0.000000 | 0.000000 | 0.321429 | 0.321429 | 0.321429 | 1.000000 | 0 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.678571 | 1.000000 | 16.000000 | 0.063245 | 1.000000 | 0.000000 | 0.000000 | 1.321429 |
| flow_closed_loop_commit1 | learned | True | 2 | 0.000000 | 0.000000 | 0.000000 | 0.666667 | 0.666667 | 0.666667 | 1.000000 | 0 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.333333 | 4.000000 | 64.000000 | 1.124588 | 0.583333 | 2.500000 | 0.000000 | 1.333333 |
| flow_closed_loop_commit1_no_contract_mask | learned | False | 2 | 0.000000 | 0.000000 | 0.000000 | 0.291667 | 0.291667 | 0.291667 | 1.000000 | 0 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.708333 | 7.000000 | 112.000000 | 2.695633 | 0.541667 | 5.500000 | 0.000000 | 1.291667 |
| flow_closed_loop_commit2 | learned | True | 2 | 0.000000 | 0.000000 | 0.000000 | 0.416667 | 0.416667 | 0.416667 | 1.000000 | 0 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.583333 | 1.500000 | 24.000000 | 0.230934 | 0.750000 | 0.500000 | 0.000000 | 1.416667 |
| flow_compute_matched_blind_replan | learned | True | 2 | 0.000000 | 0.000000 | 0.000000 | 0.333333 | 0.333333 | 0.333333 | 1.000000 | 0 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.666667 | 4.000000 | 64.000000 | 0.236988 | 1.000000 | 0.000000 | 0.000000 | 1.333333 |

## Interpretation

Compare commit-1 with both open-loop and compute-matched blind replanning. Report planner calls, total NFE, and latency with success; otherwise feedback and compute are confounded.
