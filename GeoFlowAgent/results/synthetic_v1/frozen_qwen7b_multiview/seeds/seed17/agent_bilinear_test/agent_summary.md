# GeoFlowAgent closed-loop evaluation (test)

## Results

| condition | stop_controller | contract_mask | tasks | task_success | goal_reached | correct_stop_rate | contract_valid_action_rate | execution_success_rate | zero_regret_action_rate | mean_regret_label_coverage | assigned_perturbation_tasks | triggered_perturbation_tasks | perturbation_trigger_rate | assigned_task_success_rate | triggered_recovery_rate | mean_regret | mean_planner_calls | mean_total_nfe | mean_latency_seconds | mean_embedding_cache_hit_rate | mean_runtime_embedding_miss_calls | mean_redundant_calls | mean_contract_valid_candidates |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| random_contract_oracle_stop | public_goal_oracle | True | 3 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 8.666667 | 0.000000 | 0.004980 | 0.000000 | 0.000000 | 0.000000 | 1.263889 |
| metric_no_contract_mask_oracle_stop | public_goal_oracle | False | 3 | 0.000000 | 0.000000 | 0.000000 | 0.104167 | 0.104167 | 0.104167 | 1.000000 | 0 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.895833 | 16.000000 | 0.000000 | 11.896428 | 0.145833 | 13.666667 | 0.000000 | 1.083333 |
| metric_closed_loop | public_goal_oracle | True | 3 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 8.666667 | 0.000000 | 0.778665 | 0.714286 | 2.000000 | 0.000000 | 1.037037 |
| flow_open_loop | learned | True | 3 | 0.000000 | 0.000000 | 0.000000 | 0.296296 | 0.296296 | 0.296296 | 1.000000 | 0 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.703704 | 1.000000 | 16.000000 | 0.065052 | 1.000000 | 0.000000 | 0.000000 | 1.296296 |
| flow_closed_loop_commit1 | learned | True | 3 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 8.666667 | 138.666667 | 3.773791 | 0.115741 | 7.666667 | 0.000000 | 1.037037 |
| flow_closed_loop_commit1_no_contract_mask | learned | False | 3 | 0.000000 | 0.000000 | 0.000000 | 0.125000 | 0.125000 | 0.125000 | 1.000000 | 0 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.875000 | 16.000000 | 256.000000 | 8.558876 | 0.062500 | 15.000000 | 0.000000 | 1.083333 |
| flow_closed_loop_commit2 | learned | True | 3 | 0.666667 | 0.666667 | 0.666667 | 0.597436 | 0.597436 | 0.597436 | 1.000000 | 0 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.402564 | 5.333333 | 85.333333 | 2.747639 | 0.422619 | 4.333333 | 0.000000 | 1.144444 |
| flow_compute_matched_blind_replan | learned | True | 3 | 0.000000 | 0.000000 | 0.000000 | 0.231481 | 0.231481 | 0.231481 | 1.000000 | 0 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.768519 | 8.666667 | 138.666667 | 0.506621 | 1.000000 | 0.000000 | 0.000000 | 1.152778 |

## Interpretation

Compare commit-1 with both open-loop and compute-matched blind replanning. Report planner calls, total NFE, and latency with success; otherwise feedback and compute are confounded.
