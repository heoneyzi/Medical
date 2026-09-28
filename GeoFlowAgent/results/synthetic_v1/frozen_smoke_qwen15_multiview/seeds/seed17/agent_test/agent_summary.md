# GeoFlowAgent closed-loop evaluation (test)

## Results

| condition | stop_controller | contract_mask | tasks | task_success | goal_reached | correct_stop_rate | contract_valid_action_rate | execution_success_rate | zero_regret_action_rate | mean_regret_label_coverage | assigned_perturbation_tasks | triggered_perturbation_tasks | perturbation_trigger_rate | assigned_task_success_rate | triggered_recovery_rate | mean_regret | mean_planner_calls | mean_total_nfe | mean_latency_seconds | mean_embedding_cache_hit_rate | mean_runtime_embedding_miss_calls | mean_redundant_calls | mean_contract_valid_candidates |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| random_contract_oracle_stop | public_goal_oracle | True | 3 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 8.666667 | 0.000000 | 0.004963 | 0.000000 | 0.000000 | 0.000000 | 1.263889 |
| metric_no_contract_mask_oracle_stop | public_goal_oracle | False | 3 | 0.333333 | 0.333333 | 0.333333 | 0.479167 | 0.479167 | 0.479167 | 1.000000 | 0 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.520833 | 13.666667 | 0.000000 | 5.976788 | 0.104167 | 12.000000 | 0.000000 | 1.182870 |
| metric_closed_loop | public_goal_oracle | True | 3 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 8.666667 | 0.000000 | 1.556402 | 0.172619 | 6.333333 | 0.000000 | 1.222222 |
| flow_open_loop | learned | True | 3 | 0.000000 | 0.000000 | 0.000000 | 0.277778 | 0.277778 | 0.277778 | 1.000000 | 0 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.722222 | 1.000000 | 16.000000 | 0.066763 | 1.000000 | 0.000000 | 0.000000 | 1.814815 |
| flow_closed_loop_commit1 | learned | True | 3 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 8.666667 | 138.666667 | 1.506321 | 0.537037 | 4.000000 | 0.000000 | 1.240741 |
| flow_closed_loop_commit1_no_contract_mask | learned | False | 3 | 0.000000 | 0.000000 | 0.000000 | 0.187500 | 0.187500 | 0.187500 | 1.000000 | 0 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.812500 | 16.000000 | 256.000000 | 4.749815 | 0.104167 | 14.333333 | 0.000000 | 1.604167 |
| flow_closed_loop_commit2 | learned | True | 3 | 0.666667 | 0.666667 | 0.666667 | 0.650000 | 0.650000 | 0.650000 | 1.000000 | 0 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.350000 | 6.666667 | 106.666667 | 1.948407 | 0.166667 | 5.666667 | 0.000000 | 1.294444 |
| flow_compute_matched_blind_replan | learned | True | 3 | 0.000000 | 0.000000 | 0.000000 | 0.115741 | 0.115741 | 0.115741 | 1.000000 | 0 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.884259 | 8.666667 | 138.666667 | 0.506252 | 1.000000 | 0.000000 | 0.000000 | 2.000000 |

## Interpretation

Compare commit-1 with both open-loop and compute-matched blind replanning. Report planner calls, total NFE, and latency with success; otherwise feedback and compute are confounded.
