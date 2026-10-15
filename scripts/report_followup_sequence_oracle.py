"""Publish audited oracle results and the Korean share report as Markdown/PDF source."""
import json
import re
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from zj_wam.io import resolve,write_json,sha256

OUT=resolve('artifacts/followup_v1/sequence_oracle')
REPORT=resolve('docs/ZJ_WAM_RESEARCH_PROGRESS_REPORT_KO.md')
read=lambda p:json.loads(p.read_text())


def replace_block(text,start,end,block):
    assert text.count(start)==text.count(end)==1
    a=text.index(start)+len(start);b=text.index(end)
    return text[:a]+'\n'+block.strip()+'\n'+text[b:]


def figures(frame):
    fig,ax=plt.subplots(figsize=(9,4))
    x=np.arange(len(frame));w=.36
    ax.bar(x-w/2,frame.best_gain*100,w,label='Best preserving candidate',color='#246e9b')
    ax.bar(x+w/2,frame.maximum_searched_gain*100,w,label='Best ignoring object constraint',color='#c78448')
    ax.axhline(10,color='#a32732',ls='--',label='Required jerk reduction (10%)')
    ax.set_xticks(x,[str(s-210000+1) for s in frame.seed])
    ax.set(xlabel='Reused development family (1–12)',ylabel='Full-episode jerk reduction (%)',
           title='Finite oracle search: preservation versus jerk reduction')
    ax.legend(fontsize=8,ncol=1,loc='upper left');ax.spines[['top','right']].set_visible(False)
    fig.tight_layout();fig.savefig(OUT/'oracle_gain_comparison.png',dpi=180);plt.close(fig)
    chosen=int(frame.loc[frame.best_gain.idxmax(),'seed']);meta=read(OUT/f'{chosen}.json')
    with np.load(OUT/f'{chosen}.npz') as z:
        ref=z['physics'][0];trace=z['physics'][meta['best_index']]
    t=np.arange(1,len(ref)+1)*.01
    pos=np.linalg.norm(trace[:,2:4]-ref[:,2:4],axis=1)
    angle=trace[:,4]-ref[:,4];rot=np.rad2deg(np.abs(np.arctan2(np.sin(angle),np.cos(angle))))
    fig,axes=plt.subplots(2,1,figsize=(9,4.4),sharex=True)
    for ax,y,label in zip(axes,[pos,rot],['Object position error (px)','Object rotation error (deg)']):
        ax.plot(t,y,color='#246e9b',lw=1);ax.axhline(2,color='#a32732',ls='--')
        ax.set_ylabel(label,fontsize=9);ax.set_ylim(0,2.15);ax.spines[['top','right']].set_visible(False)
    axes[0].set_title(f'Preserved trajectory example: family {chosen-210000+1}, gain {100*meta["best"]["jerk_gain"]:.2f}%')
    axes[1].set_xlabel('Physical time (s)');fig.tight_layout()
    fig.savefig(OUT/'oracle_preserved_example.png',dpi=180);plt.close(fig)


def latex_source(markdown):
    escapes={'\\':r'\textbackslash{}','&':r'\&','%':r'\%','$':r'\$','#':r'\#','_':r'\_','{':r'\{','}':r'\}','~':r'\textasciitilde{}','^':r'\textasciicircum{}'}
    def esc(s):return ''.join(escapes.get(c,c) for c in s)
    def inline(s):
        pattern=r'(\[[^\]]+\]\([^)]+\)|\*\*[^*]+\*\*|`[^`]+`)'
        out=[]
        for token in re.split(pattern,s):
            m=re.fullmatch(r'\[([^\]]+)\]\(([^)]+)\)',token)
            if m:
                label,target=m.groups()
                out.append(r'\href{'+esc(target)+'}{'+esc(label)+'}')
            elif token.startswith('**') and token.endswith('**'):out.append(r'\textbf{'+esc(token[2:-2])+'}')
            elif token.startswith('`') and token.endswith('`'):out.append(r'\texttt{'+esc(token[1:-1])+'}')
            else:out.append(esc(token))
        return ''.join(out)
    lines=markdown.splitlines();body=[];i=0
    while i<len(lines):
        line=lines[i].strip()
        if line.startswith('<!--'):i+=1;continue
        if not line:body.append('');i+=1;continue
        if line.startswith('# '):
            body += [r'\begin{center}{\LARGE\bfseries '+inline(line[2:])+r'}\end{center}',r'\vspace{3mm}']
        elif line.startswith('## '):body.append(r'\section*{'+inline(line[3:])+'}')
        elif line.startswith('### '):body.append(r'\subsection*{'+inline(line[4:])+'}')
        elif line.startswith('|'):
            table=[]
            while i<len(lines) and lines[i].strip().startswith('|'):
                cells=[c.strip() for c in lines[i].strip().strip('|').split('|')]
                if not all(re.fullmatch(r'[-: ]+',c) for c in cells):table.append(cells)
                i+=1
            n=len(table[0]);width=(16.8-(n*0.24))/n
            spec=''.join(r'>{\raggedright\arraybackslash}p{'+f'{width:.3f}'+'cm}' for _ in range(n))
            body.append(r'{\small\setlength{\tabcolsep}{1.2mm}\renewcommand{\arraystretch}{1.25}')
            body.append(r'\begin{longtable}{'+spec+'}')
            body += [r'\toprule',' & '.join(r'\textbf{'+inline(c)+'}' for c in table[0])+r'\\',r'\midrule\endhead']
            for cells in table[1:]:body.append(' & '.join(inline(c) for c in cells)+r'\\')
            body += [r'\bottomrule\end{longtable}}'];continue
        elif line.startswith('!['):
            m=re.fullmatch(r'!\[([^\]]*)\]\(([^)]+)\)',line)
            path=(REPORT.parent/m.group(2)).resolve()
            body += [r'\begin{center}\includegraphics[width=\linewidth,height=.34\textheight,keepaspectratio]{'+str(path)+r'}\end{center}',
                     r'{\small '+inline(m.group(1))+'}']
        elif re.match(r'^\d+\. ',line):body.append(r'\par\noindent '+inline(line)+r'\par')
        else:body.append(inline(line))
        i+=1
    preamble=r'''\documentclass[10pt,a4paper]{article}
\usepackage[margin=21mm]{geometry}
\usepackage{fontspec}
\usepackage{kotex}
\setmainfont{DejaVu Sans}
\setmonofont{DejaVu Sans Mono}
\setmainhangulfont{UnBatang}
\setsanshangulfont{UnDotum}
\usepackage{array,longtable,booktabs,graphicx}
\usepackage{xcolor}
\usepackage{hyperref}
\hypersetup{colorlinks=true,urlcolor=blue,linkcolor=blue,pdftitle={물체 궤적을 보존하는 동작 smoothing 연구 진행 보고서}}
\setlength{\parindent}{0pt}
\setlength{\parskip}{3.5pt}
\setlength{\emergencystretch}{3em}
\linespread{1.15}
\begin{document}
'''
    title = next(line[2:] for line in markdown.splitlines() if line.startswith('# '))
    preamble = preamble.replace('pdftitle={물체 궤적을 보존하는 동작 smoothing 연구 진행 보고서}', 'pdftitle={' + esc(title) + '}')
    return preamble+'\n'.join(body)+'\n\\end{document}\n'


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--render-only', action='store_true', help='Render the edited Markdown without replacing its content or rerunning experiments')
    args = parser.parse_args()
    if args.render_only:
        REPORT.with_suffix('.tex').write_text(latex_source(REPORT.read_text()))
        print('RENDERED', REPORT.with_suffix('.tex'), flush=True)
        return
    analysis=read(OUT/'analysis.json');audit=read(OUT/'raw_audit.json');manifest=read(OUT/'manifest.json')
    assert audit['analysis_sha256']==sha256(OUT/'analysis.json')
    for path,h in manifest['inputs'].items():assert sha256(resolve(path))==h,path
    # Verify the old synthesis still points to the original analyses.
    old=read(resolve('artifacts/followup_v1/protocol/closeout_conclusion_evidence.json'))
    for item in old['verified_report_inputs']:assert sha256(resolve(item['path']))==item['sha256'],item['path']
    frame=pd.read_csv(OUT/'state_metrics.csv');figures(frame)
    summary=analysis['summary'];best=summary['best'];greedy=summary['greedy'];n=len(frame)
    count=best['useful']['count'];passed_without=int((frame.maximum_searched_gain>=.1).sum())
    table='\n'.join(f'| {int(r.seed)-210000+1} | {100*r.greedy_gain:.2f}% | {100*r.best_gain:.2f}% | {100*r.maximum_searched_gain:.2f}% | {int(r.active_blocks)} |' for r in frame.itertuples())
    block=f'''**CPU 실행과 독립 재검산을 완료했다.** {n}개 초기 family에서 중복을 제외한 {audit['candidates']:,}개 후보 실행을 기록했다. 최종 선택은 {best['preserved']}/{n}에서 물체 궤적을 보존했고 {best['active']}/{n}에서 실제 수정을 적용했지만, 전체 jerk 10% 감소까지 만족한 사례는 **{count}/{n}**이었다.

보존 조건을 만족한 최선 후보의 전체 jerk 감소는 평균 **{100*best['gain_mean']:.2f}%**, 중앙값 **{100*best['gain_median']:.2f}%**, 범위 **{100*best['gain_min']:.2f}–{100*best['gain_max']:.2f}%**였다. 순차 oracle의 평균은 {100*greedy['gain_mean']:.2f}%였다. 공동 탐색은 순차 후보를 포함하므로 평균이 낮아지지 않으며, 이 차이를 동일 예산의 공정한 성능 비교로 해석하지 않는다.

| 개발 family | 순차 oracle jerk 감소 | 최선 보존 후보 jerk 감소 | 물체 제약 무시 시 탐색된 최대 감소 | 최선 후보의 수정 구간 수 |
| --- | --- | --- | --- | --- |
{table}

![동일 후보 탐색에서 물체 보존 조건을 지킬 때와 무시할 때의 전체 jerk 감소](../artifacts/followup_v1/sequence_oracle/oracle_gain_comparison.png)

같은 탐색 후보에서 물체 제약을 무시하면 {passed_without}/{n}개 family에 jerk 10% 이상 감소 후보가 있었다. 따라서 이번 결과를 단순히 “jerk를 줄이는 후보가 전혀 없었다”로 설명할 수는 없다. 보존과 큰 jerk 감소의 동시 달성이 어려웠다. 다만 전체 연속 공간을 완전히 탐색하지 않았으므로 그 어려움의 원인이 표현 제약인지 탐색 부족인지 확정하지 않는다.

![이번 탐색에서 가장 큰 보존 조건 내 jerk 감소를 얻은 사례의 물체 위치와 회전 오차](../artifacts/followup_v1/sequence_oracle/oracle_preserved_example.png)

검증은 전체 후보의 계수 범위·다항식 공간·행동 범위·변경 금지 구간·동일 prefix, 모든 100Hz 물체 오차와 jerk 재계산, {audit['independent_coverage_samples']:,}개 제어 시점의 독립 polygon coverage 계산을 포함한다. 선택 후보와 대조 후보 {audit['fresh_replays']}개를 새로 시뮬레이션하여 저장된 궤적과 정확히 일치함을 확인했다. Nominal fallback도 선택지에 있으므로 선택 결과의 보존율 100% 자체를 일반화 가능한 안전 성능으로 해석하지 않는다.

**추가 결론:** 현재 quintic 수정 공간에서도 여러 수정을 적용하면서 물체 궤적을 보존하는 소폭 smoothing 사례는 찾았다. 그러나 이번 12개 개발 family와 유한 탐색으로는 원래 목표인 전체 jerk 10% 감소를 달성하지 못했다. 다음 구현 우선순위는 LeWM 교체보다 수정 공간·경계와 공동 탐색의 한계를 구분하는 것이다. 이 보고서 작성 중 해당 후속 확장은 실행하지 않았다.

근거: [설정과 원본 hash](../artifacts/followup_v1/sequence_oracle/manifest.json), [분석](../artifacts/followup_v1/sequence_oracle/analysis.json), [독립 재검산](../artifacts/followup_v1/sequence_oracle/raw_audit.json), [초기 상태별 결과](../artifacts/followup_v1/sequence_oracle/state_metrics.csv).
'''
    text=replace_block(REPORT.read_text(),'<!-- BEGIN SEQUENCE ORACLE SHARE RESULTS -->','<!-- END SEQUENCE ORACLE SHARE RESULTS -->',block)
    REPORT.write_text(text)
    REPORT.with_suffix('.tex').write_text(latex_source(text))
    # Preserve the previous closeout report as historical evidence; add only a new dated notice.
    historical=resolve('docs/ZJ_WAM_FOLLOWUP_RESULTS.md');history=historical.read_text()
    begin='<!-- BEGIN SEQUENCE ORACLE RESUMPTION -->';end='<!-- END SEQUENCE ORACLE RESUMPTION -->'
    notice=f'''**2026-10-12 추가 승인 범위:** 사용자의 요청으로 다중 구간 oracle 진단과 공유용 종합 보고서만 추가 수행했다. CPU에서 {n}개 기존 정책 개발 family·{audit['candidates']:,}개 후보를 실행하고 독립 감사했다. 물체 궤적 보존 내 전체 jerk 감소는 평균 {100*best['gain_mean']:.2f}%, 최대 {100*best['gain_max']:.2f}%였으며 10% 감소 동시 달성은 {count}/{n}이었다. 저장된 nominal 명령의 30초 고정 재생이며 자기 관측에 따른 정책 재계획 실험이 아니다. [공유용 상세 보고서](ZJ_WAM_RESEARCH_PROGRESS_REPORT_KO.md)와 [새 분석](../artifacts/followup_v1/sequence_oracle/analysis.json)을 참고한다. 그 밖의 확장 실험은 계속 중단 상태다. 아래 2026-10-10 종료 정리는 당시 기록으로 보존한다.'''
    if begin in history:history=replace_block(history,begin,end,notice)
    else:
        first=history.index('\n');history=history[:first+1]+'\n'+begin+'\n'+notice+'\n'+end+'\n'+history[first+1:]
    historical.write_text(history)
    write_json(OUT/'report_evidence.json',dict(report=str(REPORT.relative_to(resolve('.'))),
        report_sha256=sha256(REPORT),tex_sha256=sha256(REPORT.with_suffix('.tex')),
        analysis_sha256=sha256(OUT/'analysis.json'),raw_audit_sha256=sha256(OUT/'raw_audit.json'),
        prior_synthesis_inputs_verified=len(old['verified_report_inputs']),
        status='CPU oracle and research share document completed; PDF compilation pending; other extensions suspended'))
    print('WROTE',REPORT,flush=True)

if __name__=='__main__':main()
