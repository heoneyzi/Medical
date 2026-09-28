# GeoFlowAgent closed-loop evaluation (test)

## Results

| condition | stop_controller | contract_mask | tasks | task_success | goal_reached | correct_stop_rate | contract_valid_action_rate | execution_success_rate | zero_regret_action_rate | mean_regret_label_coverage | assigned_perturbation_tasks | triggered_perturbation_tasks | perturbation_trigger_rate | assigned_task_success_rate | triggered_recovery_rate | mean_regret | mean_planner_calls | mean_total_nfe | mean_latency_seconds | mean_embedding_cache_hit_rate | mean_runtime_embedding_miss_calls | mean_redundant_calls | mean_contract_valid_candidates |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| random_contract_oracle_stop | public_goal_oracle | True | 3 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 8.666667 | 0.000000 | 0.004864 | 0.000000 | 0.000000 | 0.000000 | 1.259259 |
| metric_no_contract_mask_oracle_stop | public_goal_oracle | False | 3 | 0.000000 | 0.000000 | 0.000000 | 0.062500 | 0.062500 | 0.062500 | 1.000000 | 0 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.937500 | 16.000000 | 0.000000 | 11.876188 | 0.083333 | 14.666667 | 0.000000 | 1.062500 |
| metric_closed_loop | public_goal_oracle | True | 3 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 8.666667 | 0.000000 | 1.755082 | 0.422619 | 4.333333 | 0.000000 | 1.000000 |
| flow_open_loop | learned | True | 3 | 0.000000 | 0.000000 | 0.000000 | 0.240741 | 0.240741 | 0.240741 | 1.000000 | 0 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.759259 | 1.000000 | 16.000000 | 0.072197 | 1.000000 | 0.000000 | 0.000000 | 1.574074 |
| flow_closed_loop_commit1 | learned | True | 3 | 0.000000 | 0.000000 | 0.000000 | 0.571429 | 0.571429 | 0.571429 | 1.000000 | 0 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.428571 | 5.000000 | 80.000000 | 1.052761 | 0.714286 | 2.000000 | 0.000000 | 1.476190 |
| flow_closed_loop_commit1_no_contract_mask | learned | False | 3 | 0.000000 | 0.000000 | 0.000000 | 0.062500 | 0.062500 | 0.062500 | 1.000000 | 0 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.937500 | 11.000000 | 176.000000 | 5.432615 | 0.416667 | 9.333333 | 0.000000 | 1.395833 |
| flow_closed_loop_commit2 | learned | True | 3 | 0.000000 | 0.000000 | 0.000000 | 0.400000 | 0.400000 | 0.400000 | 1.000000 | 0 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.600000 | 2.333333 | 37.333333 | 0.630598 | 0.800000 | 1.000000 | 0.000000 | 1.566667 |
| flow_compute_matched_blind_replan | learned | True | 3 | 0.000000 | 0.000000 | 0.000000 | 0.142857 | 0.142857 | 0.142857 | 1.000000 | 0 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.857143 | 5.000000 | 80.000000 | 0.298147 | 1.000000 | 0.000000 | 0.000000 | 1.476190 |

## Interpretation

Compare commit-1 with both open-loop and compute-matched blind replanning. Report planner calls, total NFE, and latency with success; otherwise feedback and compute are confounded.
