"""Regenerate the November 20 review manuscript and numerical tables from saved results."""
from pathlib import Path
import json, statistics as st, hashlib, shutil, csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
BASE=ROOT/'results/paper_revision_20261115'
OLD=ROOT/'papers/revision_20261116'
SC=['clean','protocol_group=known','protocol_group=unseen_combination']
COLS=['Clean','Seen Degradation','Unseen Combination']
P1=['cnn','gcn_id','hgt_id','qc_full']
P2=['hgt_id','qc_none','qc_quality','qc_consistency','input_C']
LABELS={'cnn':'CNN','gcn_id':'GCN + fixed scale/ID','hgt_id':'Native HGT + fixed scale/ID','qc_full':'Full QC-HGT (C input)','qc_none':'QC backbone, no modules','qc_quality':'QC backbone + Quality only','qc_consistency':'QC backbone + Consistency only','input_C':'Full QC-HGT (C input)'}
records={}; provenance={}
for p in sorted(BASE.glob('*/*/seed_*/test_result.json')):
    r=json.loads(p.read_text()); assert r['complete'] and len(r['evaluations'])==len(r['expected_scenarios'])
    assert set(r['evaluations'])==set(r['expected_scenarios'])
    protocol,kind,seed=p.relative_to(BASE).parts[:3]
    records[protocol,kind,r['seed']]=r
    provenance[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
assert len(records)==60
for protocol,kind,_ in records:
    assert sorted(s for pr,k,s in records if (pr,k)==(protocol,kind))==list(range(5))
def values(protocol,kind,scenario,metric='macro_f1'):
    return [records[protocol,kind,s]['evaluations'][scenario][metric] for s in range(5)]
def stats(v):return {'mean':st.mean(v),'std_ddof1':st.stdev(v),'seeds':v}
def cell(v):return f'{st.mean(v):.4f} ± {st.stdev(v):.4f}'
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(row)+' |' for row in rows])
p1={}
for kind in P1:
    rs=[records['protocol_1',kind,s]['evaluations'] for s in range(5)]
    get=lambda prefixes:[st.mean(m['macro_f1'] for n,m in r.items() if any(n.startswith(x+'=') for x in prefixes)) for r in rs]
    p1[kind]={'Clean':values('protocol_1',kind,'clean'),'Noise':get(['gaussian_noise']),'Missing':get(['random_missing','continuous_missing']),'Drift':get(['drift']),'Time Shift':get(['time_shift']),'Avg. Robustness':[st.mean(m['macro_f1'] for n,m in r.items() if n!='clean') for r in rs]}
    assert all(len(r)==21 for r in rs)
columns=list(p1[P1[0]])
t1=table(['Model']+columns,[[LABELS[k]]+[('**'+cell(p1[k][c])+'**') if k=='qc_full' else cell(p1[k][c]) for c in columns] for k in P1])
t2=table(['Model']+COLS,[[LABELS[k]]+[('**'+cell(values('protocol_2',k,sc))+'**') if k=='input_C' else cell(values('protocol_2',k,sc)) for sc in SC] for k in P2])
t4=table(['Input variant']+COLS,[[k]+[cell(values('protocol_2','input_'+k,sc)) for sc in SC] for k in 'ABCD'])
t5=table(['Test condition','Accuracy','Normal FAR','ECE','NLL'],[[name]+[cell(values('protocol_2','input_C',sc,m)) for m in ['accuracy','far','ece','nll']] for name,sc in zip(COLS,SC)])
cost={}
for k in P1:
    rs=[json.loads((BASE/'protocol_1'/k/f'seed_{s}/result.json').read_text()) for s in range(5)]
    cost[k]={'parameters':rs[0]['num_parameters'],'mean_training_validation_hours':st.mean(r['training_seconds_including_validation'] for r in rs)/3600,'actual_epochs':[r['actual_epochs'] for r in rs]}
t6=table(['P1 model','Parameters','Mean training + validation (h)','Epochs, seeds 0–4'],[[LABELS[k],f"{cost[k]['parameters']:,}",f"{cost[k]['mean_training_validation_hours']:.3f}",'/'.join(map(str,cost[k]['actual_epochs']))] for k in P1])
summary={'source_test_sha256':provenance,'p1_macro_f1':{k:{c:stats(v) for c,v in row.items()} for k,row in p1.items()},'p2':{k:{sc:{m:stats(values('protocol_2',k,sc,m)) for m in ['macro_f1','accuracy','far','ece','nll']} for sc in SC} for k in P2+['input_A','input_B','input_D']},'p1_cost':cost}
(OUT/'verified_tables.json').write_text(json.dumps(summary,indent=2))
with (OUT/'verified_tables.csv').open('w',newline='') as f:
    w=csv.writer(f,lineterminator="\n");w.writerow(['protocol','model','condition','metric','n','mean','sample_sd'])
    for k,row in p1.items():
        for c,v in row.items():w.writerow(['P1',k,c,'macro_f1',5,st.mean(v),st.stdev(v)])
    for k,row in summary['p2'].items():
        for sc,metrics in row.items():
            for m,v in metrics.items():w.writerow(['P2',k,sc,m,5,v['mean'],v['std_ddof1']])
for name in ['input_validation.png','class_validation.png']:shutil.copy2(OLD/name,OUT/name)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
fig,axes=plt.subplots(1,2,figsize=(9.2,3.6),layout='constrained')
for ax,prefix,title in zip(axes,['gaussian_noise','drift'],['(a) Gaussian noise','(b) Drift']):
    names=sorted([n for n in records['protocol_1','qc_full',0]['evaluations'] if n.startswith(prefix+'=')],key=lambda n:float(n.split('=')[1]))
    for k,color,marker in zip(P1,['#777777','#3d8f82','#cf8238','#285aa1'],['s','^','D','o']):
        vs=[values('protocol_1',k,n) for n in names]
        ax.errorbar(range(4),[st.mean(v) for v in vs],yerr=[st.stdev(v) for v in vs],label={'cnn':'CNN','gcn_id':'GCN + scale/ID','hgt_id':'HGT + scale/ID','qc_full':'Full QC-HGT'}[k],color=color,marker=marker,capsize=3,linewidth=1.5)
    ax.set_xticks(range(4),[n.split('=')[1] for n in names]);ax.set_ylim(.4,.82);ax.set_title(title);ax.set_xlabel('Degradation strength');ax.grid(alpha=.2)
axes[0].set_ylabel('Test Macro-F1');axes[1].legend(fontsize=8,loc='lower left')
fig.savefig(OUT/'test_robustness.png',dpi=220);plt.close(fig)
text=(OLD/'manuscript.md').read_text()
a=text.index('*Evidence-based revision,');b=text.index('\n\n## Abstract',a)
text=text[:a]+'*Updated review draft, 20 November 2026. The planned 60 evaluation tasks, including 40 new training runs, are complete. Tables I and II report the new five-seed held-out results; the historical November 16 draft is retained separately. This is not final camera-ready formatting.*'+text[b:]
a=text.index('\nProcess-variable graphs');b=text.index('\n\n**Index terms:',a)
text=text[:a]+'''
Process-variable graphs support fault classification when chemical process measurements are degraded. We study a quality- and consistency-aware heterogeneous graph model (QC-HGT) on the 21-class Tennessee Eastman Process extended500 benchmark. The enhanced input combines fixed training-set standardization with physical-variable and process-unit identities. Under clean training, five-seed tests yield Macro-F1 0.7542 ± 0.0276 on clean observations and 0.7161 ± 0.0254 averaged over single-degradation conditions, the highest means among the evaluated CNN, GCN, HGT, and QC-HGT configurations. Under mixed-degradation training, the full model achieves 0.7804 ± 0.0112 on clean data, 0.7729 ± 0.0143 on seen degradations, and 0.7748 ± 0.0108 on unseen combinations. The latter improves over a contemporaneous original-input control (0.4444 ± 0.0208). Controlled same-backbone ablations show a smaller incremental full-model gain of 1.34–1.39 percentage points over disabling both modules; either module alone does not improve the mean. These findings distinguish the large representation gain from the additional joint-module contribution. Accuracy remains below Macro-F1, normal-condition false alarms remain high, and the computational cost is substantially higher than that of native graph baselines. The evidence supports improved classification under the specified budget and synthetic degradations, rather than deployment readiness or superiority to untested recent methods.'''+text[b:]
text=text.replace('This revision separates three empirical questions: whether the input representation preserves useful fault information; whether reliability and consistency contribute beyond a shared backbone and input representation; and whether the resulting classifier generalizes to held-out degradation conditions. The completed input study addresses the first question. Historical tests provide a baseline for the latter questions, but do not establish superiority of the new representation on held-out data.', 'This study separates three empirical questions: whether the input representation preserves useful fault information; whether reliability and consistency contribute beyond a shared backbone and input representation; and whether the resulting classifier generalizes to held-out degradation conditions. A validation-based representation study selects candidate C before held-out inspection. Completed P1/P2 tests and same-backbone ablations then address generalization and incremental module contributions.')
text=text.replace('The study provides (i) a process graph with six directed relation types, (ii) observable-input reliability and relation-consistency mechanisms, and (iii) an audited historical evaluation plus a controlled cumulative input study. Joint-module superiority is treated as a hypothesis to test rather than an established contribution.', 'The study provides (i) a process graph with six directed relation types and an input representation preserving fixed physical identities, (ii) observable-input reliability and relation-consistency mechanisms, and (iii) five-seed clean-training and mixed-degradation evaluations with controlled input and module comparisons. The full configuration has the highest reported mean Macro-F1, but this does not establish independent benefits from each branch or statistical significance.')
text=text.replace('C is the validation-selected candidate for subsequent testing. Historical test tables below refer to the original representation, not C.', 'C was selected using validation Macro-F1 before the new test evaluations. The main test tables use this enhanced representation; A/B/D remain input-study controls.')
text=text.replace('The historical native HGT baseline has a different backbone and is not a strict no-module control for QC-HGT.', 'Native HGT has a different backbone and is an external comparator. A separate QC-backbone configuration with both modules disabled supplies the strict no-module control.')
text=text.replace('The new input study reads only training and validation observations. These are budget-limited experiments, not training on all available windows.', 'Candidate selection uses only training and validation observations, followed by separate held-out testing. The 2,000-window test cap applies to each evaluated scenario or P2 group. These are budget-limited experiments, not training on all available windows.')
text=text.replace('The new input study does not evaluate these held-out groups.', 'P2 test endpoints are Clean, Seen Degradation, and Unseen Combination; these correspond to the saved clean, known-group, and unseen-combination evaluations.')
text=text.replace('The new runs use deterministic CUDA execution on an RTX 3080 Ti. Historical runtime measurements from other hardware are not used in the new efficiency comparison.', 'All current runs use deterministic CUDA execution on an RTX 3080 Ti. Historical runtime measurements from other hardware are excluded from the current cost comparison.')
needle='Different parameter counts and early stopping can change actual computation despite shared budget limits.'
text=text.replace(needle,needle+' The completed evaluation comprises 20 existing A/B/C/D checkpoints tested under P2, 15 new P2 QC-module training runs, 20 P1 training runs, and five P2 native-HGT training runs. The existing five C checkpoints supply the full-model P2 row. Test-window selections agree across compared models within each protocol and seed; C-based graph comparisons also share training/validation windows and fitted scales. The selected clean test sample has 343 Normal windows out of 2,000, with 76–87 per fault class. Accuracy and Macro-F1 therefore need not coincide.')
a=text.index('### B. Audited historical held-out performance');b=text.index('### C. Five-seed input-representation study',a)
text=text[:a]+'''### B. Completed held-out comparisons and module ablations

Table I reports the current P1 tests. GCN and native HGT receive the same seven fixed-scale statistics, 16-dimensional variable IDs, and eight-dimensional unit IDs; unit nodes use a zero-filled variable-ID slot. Their architectures remain distinct from QC-HGT. CNN retains its temporal input and within-window normalization. Consequently, the table compares complete configurations rather than isolating architecture under identical encoders.

Within each seed, Noise, Drift, and Time Shift average four intensities, and Missing averages eight random/continuous conditions. Average Robustness is the equal-weight average of five degradation subtypes (noise, random missingness, continuous missingness, drift, and shift), each averaged over four strengths. It excludes Clean and is not the unweighted average of the four displayed degradation columns. Only after this aggregation are five-seed means and sample standard deviations calculated.

**Table I. Current P1 held-out test Macro-F1, mean ± sample SD over five seeds. Bold denotes the highest displayed mean.**

'''+t1+'''

Full QC-HGT has the highest mean in all six displayed aggregates. Its clean score exceeds identity-enhanced HGT by 1.76 percentage points and CNN by 17.26 points; its average robustness exceeds HGT by 1.65 points. These aggregate rankings do not hold at every intensity: at noise strength 0.4, CNN scores 0.5716 versus QC-HGT 0.5084, and at drift strength 0.01 native HGT scores 0.6420 versus QC-HGT 0.6309. Figure 3 makes these limits visible. Missingness and shifts leave the aggregate-statistic graph classifiers comparatively stable under the tested perturbations; this is not evidence of robustness to arbitrary outages or misalignment.

**Table II. Current P2 held-out test Macro-F1, mean ± sample SD over five seeds. All graph rows use fixed-scale and identity-enhanced inputs.**

'''+t2+'''

The full model improves over native HGT by 3.10, 3.22, and 3.32 percentage points on Clean, Seen Degradation, and Unseen Combination, respectively. The stricter comparison against the same QC backbone with both modules disabled yields gains of 1.39, 1.39, and 1.34 points. Full-model scores exceed that control for four, five, and four of the five paired seeds, respectively. Quality-only and Consistency-only both have lower means than the no-module control across all three endpoints. Thus the observed full-configuration benefit does not establish that each component independently improves classification, nor prove a synergistic interaction or statistical significance.

Tables I and II replace the historical main tables for this revised method. They are computed exclusively from completed current tests; historical original-input values remain in the archived draft and are not pooled with these observations. The same-run input-A comparison in Table IV is the appropriate control for quantifying the representation improvement.

'''+text[b:]
text=text.replace('These findings favor C for subsequent testing under the present budget;', 'These validation findings motivated selecting C before held-out testing under the present budget;')
a=text.index('### D. Class-level behavior, probability quality, and false alarms');b=text.index('## References',a)
text=text[:a]+'''### D. Held-out input gains, accuracy, and false alarms

Table IV evaluates all A/B/C/D checkpoints on the same P2 test endpoints after candidate C was fixed from validation. These are distinct from the validation values in Table III. C exceeds the original-input A control by 32.61, 32.49, and 33.04 percentage points on Clean, Seen Degradation, and Unseen Combination. The A-to-B increase remains much larger than the B-to-C increase, supporting the importance of fixed scaling and removal of input LayerNorm together; their separate contributions are not isolated. D does not exceed C in any of the three mean test scores. No test-based candidate reselection was performed.

**Table IV. P2 input-study held-out test Macro-F1, mean ± sample SD over five seeds. All variants retain both QC modules.**

'''+t4+'''

**Table V. Full QC-HGT (C input) P2 test diagnostics, mean ± sample SD over five seeds. Higher Accuracy is better; lower FAR, ECE, and NLL are better.**

'''+t5+'''

The full model's clean Accuracy is 71.44%, while its clean Macro-F1 is 78.04%; these quantify different aspects of the class distribution and must not be interchanged. FAR is the fraction of truly normal windows classified as any fault by the multiclass argmax rule, not performance at a separately calibrated alarm threshold. Full-model clean FAR is 75.28%, compared with 70.55% for original-input A, 65.54% for native HGT, and 80.17% for the no-module QC control. Thus the full model improves false alarms relative to its no-module control, but not relative to A or native HGT. Its 76.73% FAR on unseen combinations remains a substantial limitation for monitoring applications.

The validation class analysis is retained separately in Figure 2. C improves mean validation F1 for all 21 classes relative to A; the largest changes include IDV7 (2.68% to 100.00%), IDV4 (8.84% to 97.84%), and IDV5 (24.43% to 98.55%). Normal, IDV3, IDV9, and IDV15 remain difficult. These class-specific numbers are validation diagnostics and are not substituted for held-out class scores.

![Class-level validation comparison](class_validation.png)

*Figure 2. Per-class validation F1, averaged across five seeds for A and C. This retained input-study diagnostic is distinct from the held-out metrics in Tables I, II, IV, and V.*

![Held-out noise and drift tests](test_robustness.png)

*Figure 3. P1 held-out Macro-F1 under increasing noise and drift. Points and error bars represent five-seed means and sample standard deviations. Full QC-HGT leads the aggregate comparisons, but not every individual severity.*

The current full model's mean predicted sensor reliability is 0.9936, 0.9931, and 0.9930 on P2 Clean, Seen Degradation, and Unseen Combination, respectively. These near-saturated averages include unaffected sensors and do not demonstrate selective damage identification. An affected-versus-unaffected sensor analysis remains outside the completed evaluation. Classification gains alone do not establish calibrated reliability estimation or causal interpretation of the consistency branch.

### E. Computational cost and completed evaluation scope

**Table VI. Current P1 model size and training cost. Time includes validation and excludes held-out testing.**

'''+t6+'''

Full QC-HGT requires about 2.225 hours of training and validation per P1 seed, versus 0.071 hours for native HGT and 0.059 hours for CNN. The approximately 31.5-fold total-time ratio to native HGT depends on both implementation and actual epoch counts; it is not an inference-latency ratio. The input-C parameter overhead relative to input A is small (1.88%, Table III), but QC-HGT itself remains substantially more expensive than these baselines. Inference latency and memory-normalized efficiency were not benchmarked.

All 60 planned tasks completed, including held-out evaluation after each new training run. The final three P2 HGT seeds were recovered after an interruption: the incomplete seed 2 was rerun from initialization with the unchanged configuration, and seeds 3 and 4 then ran normally. Completed runs were retained, and interrupted outputs were archived. Saved records include selected windows, configuration/source fingerprints, fitted scalers, best checkpoints, and test predictions. The updated tables aggregate each seed separately and use sample standard deviations; standard deviations across severities are not averaged.

The completed model matrix contains CNN, GCN, native HGT, and QC-HGT variants. It does not contain newly implemented recent external models. Accordingly, these results support comparisons within the evaluated matrix, not a state-of-the-art claim. Additional models would require matched classes, run splits, window construction, degradation conditions, and comparable tuning and training budgets.

### F. Limitations

Evidence is restricted to one simulated process dataset, synthetic degradation rules, and a small fixed window budget. The test set was held out during candidate selection, but broader datasets and real sensor failures remain untested. Matching sampled windows and graph inputs does not eliminate architecture, initialization, compute, and early-stopping differences. CNN uses a different temporal representation and normalization. The full model's advantage is a descriptive mean comparison over five seeds; no significance test is claimed, and overlapping windows must not be treated as independent observations in future uncertainty analyses. Standalone modules fail to improve the no-module mean, severe noise/drift can reverse rankings, normal-condition FAR remains high, and QC-HGT has a substantial training-cost penalty. These constraints prevent claims of uniformly improved monitoring, calibrated sensor-health estimation, or deployment readiness.

## IV. Conclusions

Fixed training-set scaling and physical identities substantially improve QC-HGT's fault classification in the completed TEP experiments. Under clean training, enhanced QC-HGT reaches clean Macro-F1 0.7542 ± 0.0276 and average robustness 0.7161 ± 0.0254, the highest aggregate means among the evaluated models. Under mixed-degradation training, it achieves 0.7804 ± 0.0112, 0.7729 ± 0.0143, and 0.7748 ± 0.0108 on Clean, Seen Degradation, and Unseen Combination. Same-backbone ablations separate the large representation improvement from the smaller full-module gain of approximately 1.3–1.4 percentage points. Neither standalone module improves the no-module mean, so independent component benefits are not established. High false-alarm rates, severity-dependent rankings, and substantial computational cost remain unresolved. Comparisons with recent external models and practical safety-risk validation require additional evidence.

'''+text[b:]
(OUT/'manuscript.md').write_text(text)
print('Generated manuscript, tables and test robustness figure:',OUT)
