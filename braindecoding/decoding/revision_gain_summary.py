"""Current v2 A/C performance report from the completed fixed preview."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

from braindecoding.config import PROJECT_ROOT
from braindecoding.unified_sentence_v2 import read, save, sha256
from braindecoding.unified_sentence_v2_preview import PREVIEW_OUT, REPORT as PREVIEW_REPORT, plan as preview_plan
from braindecoding.unified_sentence_v2_results import METRICS, metric_row
from braindecoding.unified_sentence_v2_artifacts import csv_write

CONFIG = PROJECT_ROOT / "configs/decoding/revision_gain_preview.yaml"
REPORT = PROJECT_ROOT / "data/results/studies/unified_sentence_v2_revision_gain/results.json"
NAMES = {"pallier2025":"Pallier2025", "chineseeeg2_littleprince":"ChineseEEG-2"}
CONDITIONS = ("A", "C")


def profile():
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    if config["decoding"]["conditions"] != list(CONDITIONS) or config["decoding"]["alignment"] != "real" or config["decoding"]["branch"] != "open" or config["controls"] or config["ablations"]:
        raise ValueError("Current profile requires only real/open A and C")
    if config["population"]["scope"] != "fixed_preview" or config["population"]["seeds"] != [0]:
        raise ValueError("Material/seed expansion requires a separate user decision")
    if config["execution"]["test_evaluation_allowed"] or config["calibration"]["recalibrate"]:
        raise ValueError("Current report reuses the validation calibration")
    return config


def transitions(group, result):
    """Distinguish A/C final improvement and C's own online repair actions."""
    ref = group["reference"]
    n = len(ref)
    if any(len(result["final"][c]) != n for c in CONDITIONS) or len(result["trace"]) != n:
        raise ValueError("Known word count or arrival population changed")
    initial = []
    for pos,step in enumerate(result["trace"]):
        if step["arrival"] != pos or len(step["C"]) != pos+1:
            raise ValueError("C arrival trace differs from known boundaries")
        initial.append(step["C"][pos])
    a,c = result["final"]["A"],result["final"]["C"]
    return dict(subject=group["subject"],group=group["group"],words=n,eligible=n-1,
        paired_repaired=sum(x!=t and y==t for t,x,y in zip(ref,a,c)),
        paired_harmed=sum(x==t and y!=t for t,x,y in zip(ref,a,c)),
        rolling_repaired=sum(initial[i]!=ref[i] and c[i]==ref[i] for i in range(n-1)),
        rolling_harmed=sum(initial[i]==ref[i] and c[i]!=ref[i] for i in range(n-1)))


def metric_rows(preview,dataset,decoded):
    groups={g["group"]:g for g in preview["datasets"][dataset]["groups"]["evaluation"]}
    records=decoded["records"]
    if decoded["preview_sha256"] != preview["preview_sha256"] or decoded["test_evaluated"] is not False:
        raise ValueError("Cached preview provenance differs")
    if len(records) != len(groups) or {r["group"] for r in records} != set(groups):
        raise ValueError("Cached evaluation population incomplete or duplicated")
    rows=[]
    for record in records:
        g=groups[record["group"]]
        transitions(g,record["result"])
        for c in CONDITIONS:
            rows.append(dict(dataset=dataset,seed=0,subject=g["subject"],group=g["group"],material=g["material"],block=g["block"],scope=g["scope"],condition=c,words=len(g["reference"]),**metric_row(g["reference"],record["result"]["final"][c],preview["datasets"][dataset]["language"]=="zh")))
    return rows


def subject_summary(rows):
    cells = defaultdict(list)
    for row in rows:
        if row["condition"] in CONDITIONS:
            cells[(row["subject"],row["condition"])].append(row)
    subjects = []
    for (subject,condition),members in sorted(cells.items()):
        result = dict(subject=subject,condition=condition,segments=len(members),words=sum(r["words"] for r in members))
        for metric in METRICS:
            result[metric] = float(np.mean([r[metric] for r in members])) if metric=="sentence_bleu1_mean" else float(np.average([r[metric] for r in members],weights=[r["words"] for r in members]))
        subjects.append(result)
    summary = [dict(condition=c,**{m:float(np.mean([r[m] for r in subjects if r["condition"]==c])) for m in METRICS}) for c in CONDITIONS]
    return summary,subjects


def revision_summary(rows):
    by_subject = defaultdict(list)
    for row in rows: by_subject[row["subject"]].append(row)
    output = {k:sum(r[k] for r in rows) for k in ("words","eligible","paired_repaired","paired_harmed","rolling_repaired","rolling_harmed")}
    for kind,denominator in (("paired","words"),("rolling","eligible")):
        for outcome in ("repaired","harmed"):
            key = f"{kind}_{outcome}"
            output[key+"_rate"] = float(np.mean([sum(r[key] for r in members)/sum(r[denominator] for r in members) if sum(r[denominator] for r in members) else 0. for members in by_subject.values()]))
    output["net_correct_words"] = output["paired_repaired"]-output["paired_harmed"]
    return output


def export():
    config = profile()
    preview,parent = preview_plan()
    original = read(PREVIEW_REPORT)
    if original["status"] != "completed_preview" or original["preview_sha256"] != preview["preview_sha256"]:
        raise ValueError("Fixed real-brain preview must be complete")
    datasets,by_group,by_subject,repair_rows,sources = {},[],[],[],[]
    for dataset,info in preview["datasets"].items():
        path = PREVIEW_OUT/dataset/"main.json"
        decoded = read(path)
        if decoded["seed"] != 0 or decoded["selected_weight"] != config["calibration"]["selected_language_weights"][dataset]:
            raise ValueError("Existing seed/calibration differs from current profile")
        rows=metric_rows(preview,dataset,decoded)
        summary,subjects = subject_summary(rows)
        lookup = {r["condition"]:r for r in summary}
        groups = {g["group"]:g for g in info["groups"]["evaluation"]}
        changes = [dict(dataset=dataset,**transitions(groups[r["group"]],r["result"])) for r in decoded["records"]]
        gains = dict(sentence_bleu1=lookup["C"]["sentence_bleu1_mean"]-lookup["A"]["sentence_bleu1_mean"],aligned_word_accuracy=lookup["C"]["aligned_word_accuracy"]-lookup["A"]["aligned_word_accuracy"],wer_reduction=lookup["A"]["wer"]-lookup["C"]["wer"])
        revisions = revision_summary(changes)
        for c in CONDITIONS:
            expected = next(r for r in original["datasets"][dataset]["summary"] if r["condition"]==c)
            if any(not np.isclose(lookup[c][m],expected[m],atol=1e-12,rtol=0.) for m in METRICS):
                raise ValueError("A/C metric differs from completed source")
        source_repair = next(r for r in original["datasets"][dataset]["repairs"] if r["condition"]=="C")
        if revisions["rolling_repaired"] != source_repair["repaired"] or revisions["rolling_harmed"] != source_repair["harmed"]:
            raise ValueError("Online repair count differs from source")
        examples = original["datasets"][dataset]["examples"]
        for example in examples:
            example["final"] = {c:example["final"][c] for c in CONDITIONS}
            example["trace"] = [{k:v for k,v in step.items() if k in ("arrival",*CONDITIONS)} for step in example["trace"]]
            for key in list(example):
                if key.endswith("_B"): del example[key]
        datasets[dataset] = dict(scope=info["counts"]["evaluation"],selected_weight=decoded["selected_weight"],summary=summary,gains=gains,revisions=revisions,by_subject=subjects,examples=examples)
        by_group.extend(rows)
        by_subject.extend(dict(dataset=dataset,**r) for r in subjects)
        repair_rows.extend(changes)
        sources.append(dict(path=path.relative_to(PROJECT_ROOT).as_posix(),sha256=sha256(path)))
    output = dict(status="completed_revision_gain_preview",profile=config["profile"]["id"],profile_sha256=sha256(CONFIG),preview_sha256=preview["preview_sha256"],parent_manifest_sha256=parent["manifest_sha256"],conditions=list(CONDITIONS),branch="open",alignment="real",comparison="C_minus_A",evidence="descriptive_development_preview",seeds=[0],datasets=datasets,sources=sources,test_evaluated=False,full_experiment_completed=False,metric_contract="BLEU1: within-subject segment mean then equal subjects; word accuracy/WER pool words within subjects before equal subjects.",interpretation="Overall delayed rolling improvement, with first-to-final online repair counts.")
    save(REPORT,output)
    prefix = PROJECT_ROOT/"data/results/exports/unified_sentence_v2_revision_gain_"
    for suffix,rows in (("by_group",by_group),("by_subject",by_subject),("repairs",repair_rows)):
        csv_write(Path(str(prefix)+suffix+".csv"),rows)
    render(output)
    markdown(output)
    return output


def render(report):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"pdf.fonttype":42,"svg.fonttype":"none","axes.spines.top":False,"axes.spines.right":False})
    fig,axes = plt.subplots(1,2,figsize=(9.6,3.8),sharey=True)
    limit = max(r["sentence_bleu1_mean"] for d in report["datasets"].values() for r in d["summary"])*135
    for ax,(dataset,info) in zip(axes,report["datasets"].items()):
        values = [r["sentence_bleu1_mean"]*100 for r in info["summary"]]
        ax.bar(range(2),values,color=["#8C9AA5","#2C8190"],width=.55)
        ax.set_xticks(range(2),["A: fixed","C: rolling correction"])
        ax.set_ylim(0,limit)
        ax.set_title(f"{NAMES[dataset]}\n{info['scope']['subjects']} subjects, 6 materials")
        for x,v in enumerate(values): ax.text(x,v+limit*.025,f"{v:.2f}",ha="center",fontsize=9)
        ax.text(.5,.95,f"C - A = {info['gains']['sentence_bleu1']*100:+.2f} percentage points",transform=ax.transAxes,ha="center",va="top",fontsize=9)
    axes[0].set_ylabel("Mean segment BLEU-1 (%)")
    fig.suptitle("v2 delayed rolling revision | fixed preview | seed 0",fontsize=11)
    fig.tight_layout()
    destination = PROJECT_ROOT/"data/results/figures/unified_sentence_v2_revision_gain"
    destination.mkdir(parents=True,exist_ok=True)
    files = []
    for suffix in ("png","pdf","svg"):
        path = destination/f"revision_gain.{suffix}"
        fig.savefig(path,dpi=180,bbox_inches="tight")
        files.append(dict(path=path.relative_to(PROJECT_ROOT).as_posix(),sha256=sha256(path)))
    plt.close(fig)
    save(destination/"sources.json",dict(report_sha256=sha256(REPORT),implementation_sha256=sha256(__file__),files=files,shared_y_axis=True,conditions=list(CONDITIONS)))


def markdown(report):
    lines = ["# v2 句子解码：延迟滚动修正的性能提升", "",
        "当前主实验比较真实神经输入下的 A（固定延迟输出）与 C（延迟滚动修正），主比较为 C−A。",
        "C 每次重新搜索最近三个词，旧词使用当前可用的 K1/K2；更早输出固定。A 沿用 K0，输出后固定。",
        "模型、完整训练词库、开放补词、Top-20、束宽 5、完整词评分和语言模型沿用冻结 v2 实现。两个数据集均使用已有独立校准子集所选权重 0.1。", "",
        "## 当前主结果", "", "固定 seed 0、每个数据集 6 个评价材料，保留全部受试者。主指标为受试者等权的平均片段 BLEU-1（词重合指标）；当前为材料子集上的描述性开发验证。", "",
        "| 数据集 | A BLEU-1 (%) | C BLEU-1 (%) | C−A（百分点） | A 对齐词准确率 (%) | C 对齐词准确率 (%) | WER 降低（百分点） |", "|---|---:|---:|---:|---:|---:|---:|"]
    for dataset,info in report["datasets"].items():
        s={r["condition"]:r for r in info["summary"]}
        lines.append(f"| {NAMES[dataset]} | {s['A']['sentence_bleu1_mean']*100:.2f} | {s['C']['sentence_bleu1_mean']*100:.2f} | {info['gains']['sentence_bleu1']*100:+.2f} | {s['A']['aligned_word_accuracy']*100:.2f} | {s['C']['aligned_word_accuracy']*100:.2f} | {info['gains']['wer_reduction']*100:.2f} |")
    lines += ["", "![A 与 C 的主比较](../../figures/unified_sentence_v2_revision_gain/revision_gain.png)", "", "## 最终输出的改善", "", "相同词位的 A 最终输出与 C 最终输出比较，覆盖全部词位。净增正确词 = 改错为对−改对为错。", "", "| 数据集 | 评价词位 | A→C 改错为对 | A→C 改对为错 | 净增正确词 |", "|---|---:|---:|---:|---:|"]
    for dataset,info in report["datasets"].items():
        r=info["revisions"]
        lines.append(f"| {NAMES[dataset]} | {r['words']} | {r['paired_repaired']} | {r['paired_harmed']} | {r['net_correct_words']} |")
    lines += ["", "## 滚动过程中的回修动作", "", "C 自身的首次输出与最终输出比较。只有片段末词之前的位置计入可修正词位，末词没有片段内后续到达事件。此定义与上表 A→C 比较不同。", "", "| 数据集 | 可修正词位 | 改对 | 误改 | 改对率 (%) | 误改率 (%) |", "|---|---:|---:|---:|---:|---:|"]
    for dataset,info in report["datasets"].items():
        r=info["revisions"]
        lines.append(f"| {NAMES[dataset]} | {r['eligible']} | {r['rolling_repaired']} | {r['rolling_harmed']} | {r['rolling_repaired_rate']*100:.2f} | {r['rolling_harmed_rate']*100:.2f} |")
    lines += ["", "## 真实回修例子", "", "复用原预览中既定展示规则选出的‘改对最多’案例，仅用于展示过程。到达事件按词计，不代表等长的墙钟时间。", ""]
    for dataset,info in report["datasets"].items():
        example=next(e for e in info["examples"] if e["selection"]=="most_C_repairs")
        lines += [f"### {NAMES[dataset]} · {example['subject']}", "", "| 位置 | 参考词 | A 最终 | C 首次 | 后 1 词到达 | 后 2 词到达 | C 最终 |", "|---:|---|---|---|---|---|---|"]
        for pos in example["repaired_C"][:3]:
            trace=example["trace"]
            after1=trace[pos+1]["C"][pos] if pos+1<len(trace) else "—"
            after2=trace[pos+2]["C"][pos] if pos+2<len(trace) else "—"
            cells=[pos+1,example["reference"][pos],example["final"]["A"][pos],example["initial_C"][pos],after1,after2,example["final"]["C"][pos]]
            lines.append("| "+" | ".join(str(c).replace("|","\\|").replace("\n"," ") for c in cells)+" |")
        lines.append("")
    lines += ["## 当前协议", "", "主比较衡量整个延迟滚动方法相对固定输出的收益。词数与材料边界已知。后续扩大评价范围时，主条件继续为 A/C，沿用同样的评分和统计定义。", "", "当前全量运行仍等待用户决定，测试集保持封存。执行配置：`configs/decoding/revision_gain_preview.yaml`；协议说明：`data/manifests/rolling_sentence/unified_protocol_v2.md`。", "", f"原固定预览 SHA-256：`{report['preview_sha256']}`。", ""]
    REPORT.with_suffix(".md").write_text("\n".join(lines),encoding="utf-8")


def main():
    report=export()
    print(dict(status=report["status"],comparison=report["comparison"],report=str(REPORT.with_suffix(".md"))))


if __name__=="__main__":
    main()
