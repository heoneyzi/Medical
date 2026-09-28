# GeoFlowAgent closed-loop evaluation (test)

## Results

| condition | stop_controller | contract_mask | tasks | task_success | goal_reached | correct_stop_rate | contract_valid_action_rate | execution_success_rate | zero_regret_action_rate | mean_regret_label_coverage | assigned_perturbation_tasks | triggered_perturbation_tasks | perturbation_trigger_rate | assigned_task_success_rate | triggered_recovery_rate | mean_regret | mean_planner_calls | mean_total_nfe | mean_latency_seconds | mean_embedding_cache_hit_rate | mean_runtime_embedding_miss_calls | mean_redundant_calls | mean_contract_valid_candidates |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| random_contract_oracle_stop | public_goal_oracle | True | 3 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 8.666667 | 0.000000 | 0.004334 | 0.000000 | 0.000000 | 0.000000 | 1.115741 |
| metric_no_contract_mask_oracle_stop | public_goal_oracle | False | 3 | 0.000000 | 0.000000 | 0.000000 | 0.187500 | 0.187500 | 0.187500 | 1.000000 | 0 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.812500 | 16.000000 | 0.000000 | 12.536298 | 0.145833 | 13.666667 | 0.000000 | 1.062500 |
| metric_closed_loop | public_goal_oracle | True | 3 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 8.666667 | 0.000000 | 1.745796 | 0.422619 | 4.333333 | 0.000000 | 1.000000 |
| flow_open_loop | learned | True | 3 | 0.000000 | 0.000000 | 0.000000 | 0.200000 | 0.200000 | 0.200000 | 1.000000 | 0 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.800000 | 1.000000 | 16.000000 | 0.067424 | 1.000000 | 0.000000 | 0.000000 | 1.466667 |
| flow_closed_loop_commit1 | learned | True | 3 | 0.333333 | 0.333333 | 0.333333 | 0.500000 | 0.500000 | 0.500000 | 1.000000 | 0 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.500000 | 4.000000 | 64.000000 | 1.408187 | 0.703704 | 2.666667 | 0.000000 | 1.500000 |
| flow_closed_loop_commit1_no_contract_mask | learned | False | 3 | 0.000000 | 0.000000 | 0.000000 | 0.233333 | 0.233333 | 0.233333 | 1.000000 | 0 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.766667 | 4.333333 | 69.333333 | 1.446231 | 0.700000 | 3.000000 | 0.000000 | 1.533333 |
| flow_closed_loop_commit2 | learned | True | 3 | 0.333333 | 0.333333 | 0.333333 | 0.383838 | 0.383838 | 0.383838 | 1.000000 | 0 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.616162 | 3.000000 | 48.000000 | 1.102704 | 0.555556 | 2.000000 | 0.000000 | 1.444444 |
| flow_compute_matched_blind_replan | learned | True | 3 | 0.000000 | 0.000000 | 0.000000 | 0.261905 | 0.261905 | 0.261905 | 1.000000 | 0 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.738095 | 4.000000 | 64.000000 | 0.241968 | 1.000000 | 0.000000 | 0.000000 | 1.547619 |

## Interpretation

Compare commit-1 with both open-loop and compute-matched blind replanning. Report planner calls, total NFE, and latency with success; otherwise feedback and compute are confounded.
