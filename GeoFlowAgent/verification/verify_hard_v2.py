"""Recompute the published hard-v2 aggregates from preserved JSON. No training."""
import argparse
import json
from pathlib import Path
from statistics import mean

def verify(root):
    reports = root / 'final_reports'  # portfolio copy: flattened from tmp/GeoFlowAgent_a_runs/search_hard_v2/final/reports
    data = json.loads((reports/'value_test/summary.json').read_text(encoding='utf-8'))
    selected = [r for r in data['runs'] if r['seed'] in [17, 29, 43]]
    reference = [r for r in data['runs'] if r['seed'] == 19][0]
    assert len(selected) == 3
    assert all(r['evaluation_split'] == 'test' and r['metrics']['task_count'] == 48 for r in data['runs'])
    metrics = ['policy_optimal_set_accuracy', 'joint_stop_action_accuracy', 'regret_at_1']
    aggregate = {k: mean(r['metrics'][k] for r in selected) for k in metrics}
    for k, expected in zip(metrics, [0.8010, 0.7364, 0.1403]):
        assert abs(aggregate[k]-expected) < .00005, (k,aggregate[k],expected)
    paired = {}
    for key in ['per_task_policy_accuracy','per_task_joint_accuracy','per_task_regret_at_1']:
        tasks = set(reference['metrics'][key])
        assert len(tasks) == 48
        assert all(set(r['metrics'][key]) == tasks for r in selected)
        paired[key] = mean(mean(r['metrics'][key][t] for r in selected)-reference['metrics'][key][t] for t in sorted(tasks))
    for k, expected in zip(paired,[.1198,.0691,-.2489]):
        assert abs(paired[k]-expected) < .00005
    arms = ['open_loop','receding_horizon','compute_matched_blind_replanning','receding_horizon_verifier_stop_guard']
    flows = [json.loads((reports/f'state_flow_seed{s}_test.json').read_text(encoding='utf-8')) for s in [17,29,43]]
    assert all(d['metrics']['eligible_tasks'] == 48 and d['evaluation_scope'] == 'task_roots' for d in flows)
    flow_mean = {arm:mean(d['metrics'][arm]['goal_completion_rate'] for d in flows) for arm in arms}
    for arm, expected in zip(arms,[.1233,.4028,0.,.4375]):
        assert abs(flow_mean[arm]-expected) < .00005
    result = {'status':'PASS','selected_seeds':[17,29,43], 'reference_seed':19,'test_tasks':48,'state_metric_seed_means':aggregate,'reference_state_metrics':{k:reference['metrics'][k] for k in metrics},'paired_task_macro_differences':paired,'flow_seed_means':flow_mean,'scope':'Recomputed point estimates from saved metrics; historical confidence intervals and GPU training not rerun.'}
    return result

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',nargs='?',type=Path,default=Path(__file__).resolve().parents[1]/'results/hard_v2')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args(); result=verify(args.root)
    text=json.dumps(result,ensure_ascii=False,indent=2)
    if args.output:args.output.write_text(text+'\n',encoding='utf-8')
    print(text)
