# GeoFlowAgent closed-loop evaluation (test)

## Results

| condition | stop_controller | contract_mask | tasks | task_success | goal_reached | correct_stop_rate | contract_valid_action_rate | execution_success_rate | zero_regret_action_rate | mean_regret_label_coverage | assigned_perturbation_tasks | triggered_perturbation_tasks | perturbation_trigger_rate | assigned_task_success_rate | triggered_recovery_rate | mean_regret | mean_planner_calls | mean_total_nfe | mean_latency_seconds | mean_embedding_cache_hit_rate | mean_runtime_embedding_miss_calls | mean_redundant_calls | mean_contract_valid_candidates |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| random_contract_oracle_stop | public_goal_oracle | True | 3 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 8.666667 | 0.000000 | 0.004513 | 0.000000 | 0.000000 | 0.000000 | 1.115741 |
| metric_no_contract_mask_oracle_stop | public_goal_oracle | False | 3 | 0.000000 | 0.000000 | 0.000000 | 0.125000 | 0.125000 | 0.125000 | 1.000000 | 0 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.875000 | 16.000000 | 0.000000 | 6.878475 | 0.083333 | 14.666667 | 0.000000 | 1.062500 |
| metric_closed_loop | public_goal_oracle | True | 3 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 8.666667 | 0.000000 | 1.083253 | 0.422619 | 4.333333 | 0.000000 | 1.000000 |
| flow_open_loop | learned | True | 3 | 0.000000 | 0.000000 | 0.000000 | 0.203571 | 0.203571 | 0.203571 | 1.000000 | 0 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.796429 | 1.000000 | 16.000000 | 0.068251 | 1.000000 | 0.000000 | 0.000000 | 1.952381 |
| flow_closed_loop_commit1 | learned | True | 3 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 8.666667 | 138.666667 | 1.652930 | 0.495370 | 4.333333 | 0.000000 | 1.157407 |
| flow_closed_loop_commit1_no_contract_mask | learned | False | 3 | 0.000000 | 0.000000 | 0.000000 | 0.175000 | 0.175000 | 0.175000 | 1.000000 | 0 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.825000 | 15.666667 | 250.666667 | 4.723098 | 0.106944 | 14.000000 | 0.000000 | 1.822222 |
| flow_closed_loop_commit2 | learned | True | 3 | 0.666667 | 0.666667 | 0.666667 | 0.669872 | 0.669872 | 0.669872 | 1.000000 | 0 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.330128 | 6.000000 | 96.000000 | 1.686786 | 0.178571 | 5.000000 | 0.000000 | 1.285256 |
| flow_compute_matched_blind_replan | learned | True | 3 | 0.000000 | 0.000000 | 0.000000 | 0.115741 | 0.115741 | 0.115741 | 1.000000 | 0 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.884259 | 8.666667 | 138.666667 | 0.519985 | 1.000000 | 0.000000 | 0.000000 | 2.000000 |

## Interpretation

Compare commit-1 with both open-loop and compute-matched blind replanning. Report planner calls, total NFE, and latency with success; otherwise feedback and compute are confounded.
