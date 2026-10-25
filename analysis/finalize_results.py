"""Verify saved models, refresh TF annotation from frozen Lambert list, write report.
Run after run_analysis.py: python analysis/finalize_results.py --data .
"""
import argparse,hashlib,json,html,re,importlib.metadata
from pathlib import Path
import numpy as np,pandas as pd,h5py
from scipy.stats import pearsonr
from sklearn.metrics import r2_score,mean_squared_error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True);a=ap.parse_args()
R=Path(__file__).resolve().parent
def save(d,n,index=False):d.to_csv(R/n,index=index)
def fig(n):
 for ext in ['png','pdf']:plt.savefig(R/'figures'/f'{n}.{ext}',dpi=170,bbox_inches='tight')
 plt.close()
def table(d):
 d=d.copy()
 for c in d:
  if pd.api.types.is_float_dtype(d[c]):d[c]=d[c].map(lambda v:f'{v:.3f}')
 return '| '+' | '.join(map(str,d.columns))+' |\n| '+' | '.join(['---']*len(d.columns))+' |\n'+'\n'.join('| '+' | '.join(str(v).replace('|','/') for v in row)+' |' for row in d.itertuples(index=False,name=None))
focus=['erythrocyte','platelet','monocyte','B cell','CD4-positive, alpha-beta T cell','natural killer cell']
cn={'erythrocyte':'红细胞','platelet':'血小板','monocyte':'单核细胞','B cell':'B细胞','CD4-positive, alpha-beta T cell':'CD4 T细胞','natural killer cell':'NK细胞'}
short={'hematopoietic stem cell':'HSC','common myeloid progenitor':'CMP','erythrocyte':'Erythrocyte','platelet':'Platelet','macrophage':'Macrophage','B cell':'B cell','monocyte':'Monocyte','natural killer cell':'NK cell','CD4-positive, alpha-beta T cell':'CD4 T','CD8-positive, alpha-beta T cell':'CD8 T','basophil':'Basophil','neutrophil':'Neutrophil','myeloid dendritic cell':'Myeloid DC','plasmacytoid dendritic cell':'Plasmacytoid DC','plasma cell':'Plasma cell','mature NK T cell':'NKT','regulatory T cell':'Treg','classical monocyte':'Classical mono','non-classical monocyte':'Nonclassical mono','naive thymus-derived CD4-positive, alpha-beta T cell':'Naive CD4 T','intermediate monocyte':'Intermediate mono','hematopoietic precursor cell':'Precursor'}
tfs=set((R/'human_tfs.txt').read_text().splitlines());ac=pd.read_csv(R/'models/all_perturbation_coefficients.csv');ac['is_TF']=ac.perturbation.isin(tfs)
tf=ac[ac.is_TF&ac.selected].copy();cells=list(ac.cell_type.unique());save(ac,'models/all_perturbation_coefficients.csv');save(tf,'models/selected_TFs.csv');save(tf.groupby('cell_type',sort=False).head(10),'models/top10_TFs.csv');save(ac[ac.selected].groupby('cell_type',sort=False).head(20),'models/top20_perturbations.csv')
m=pd.read_csv(R/'models/test_metrics.csv');m['n_TF_nonzero']=m.cell_type.map(tf.groupby('cell_type').size()).fillna(0).astype(int);save(m,'models/test_metrics.csv')
b=pd.read_csv(a.data/'blood mean expr.csv',index_col=0);b.index=b.index.str.strip();b=b[~b.index.duplicated()]
with h5py.File(a.data/'L1000 MCF7_OE gene expression.h5') as f:x=pd.DataFrame(f['expression/block0_values'][:],index=[z.decode().strip() for z in f['expression/axis1'][:]],columns=[z.decode() for z in f['expression/axis0'][:]])
x=x[~x.index.duplicated()];cm=pd.read_csv(R/'models/coefficient_matrix_standardized.csv',index_col=0);names=cm.index
annotation=pd.DataFrame({'perturbation':x.columns,'is_TF':x.columns.isin(tfs)});save(annotation,'qc/perturbation_TF_annotation.csv')
canonical=['GATA1','GATA2','SPI1','CEBPA','RUNX1','TAL1','KLF1','MYB','PAX5','BCL11A','TCF7']
coverage=pd.DataFrame({'TF':canonical,'available_as_L1000_perturbation':[v in x.columns for v in canonical],'measured_as_response_gene':[v in x.index for v in canonical]});save(coverage,'qc/canonical_TF_coverage.csv')
split=pd.read_csv(R/'models/gene_split.csv');tr=split.loc[split.split=='train','gene_symbol'];te=split.loc[split.split=='test','gene_symbol'];assert len(set(tr)&set(te))==0;assert split.gene_symbol.is_unique
trainX=x.loc[tr,names].to_numpy(dtype=float);xm=trainX.mean(axis=0);xs=trainX.std(axis=0);xs=np.where(xs>1e-12,xs,1)
y=b.loc[tr,cells].sub(b.loc[tr,'hematopoietic stem cell'],axis=0);ym=y.mean().to_numpy();ys=y.std(ddof=0).to_numpy()
orig=cm.to_numpy()*ys[None,:]/xs[:,None];intercept=ym-xm@orig
save(pd.DataFrame({'perturbation':names,'training_mean':xm,'training_SD':xs}),'models/feature_scaler.csv')
save(pd.DataFrame({'cell_type':cells,'response_training_mean':ym,'response_training_SD':ys,'intercept_original_X':intercept}),'models/model_parameters.csv')
held=pd.read_csv(R/'models/heldout_gene_predictions.csv.gz');checks=[]
for j,c in enumerate(cells):
 d=held[held.cell_type==c];assert set(d.gene_symbol)==set(te)
 recon=x.loc[d.gene_symbol,names].to_numpy()@orig[:,j]+intercept[j]
 err=float(np.max(np.abs(recon-d.predicted_delta.to_numpy())));assert err<1e-8,(c,err)
 truth=b.loc[d.gene_symbol,c].to_numpy()-b.loc[d.gene_symbol,'hematopoietic stem cell'].to_numpy();assert np.allclose(truth,d.observed_delta)
 score=r2_score(truth,recon);assert abs(score-m.set_index('cell_type').loc[c,'test_R2'])<1e-10
 checks.append({'cell_type':c,'max_prediction_reconstruction_error':err,'recomputed_R2':score,'passed':True})
save(pd.DataFrame(checks),'models/verification.csv')
dg=pd.read_csv(R/'models/convergence_diagnostics.csv');assert dg.max_dual_gap.max()<1.001e-4;assert dg.max_iterations.max()<30000
assert set(tf.perturbation)<=tfs;assert not {'RIOK2','DNMT3A','HDAC1','HMGB1','PML'}&set(tf.perturbation)
assert np.isfinite(ac.coef_standardized).all();assert ac.selection_frequency_5_subsamples.between(0,1).all()
sets={c:set(tf[tf.cell_type==c].head(10).perturbation) for c in cells}
jac=pd.DataFrame([[len(sets[c]&sets[d])/max(len(sets[c]|sets[d]),1) for d in cells] for c in cells],index=cells,columns=cells);save(jac,'models/top10_TF_Jaccard.csv',True)
ov=[]
for i,c in enumerate(cells):
 for d in cells[i+1:]:ov.append({'cell_a':c,'cell_b':d,'n_shared':len(sets[c]&sets[d]),'Jaccard':jac.loc[c,d],'shared_TFs':';'.join(sorted(sets[c]&sets[d]))})
ov=pd.DataFrame(ov);save(ov,'models/TF_pairwise_overlap.csv')
tg=tf.groupby('perturbation').agg(n_cell_types=('cell_type','nunique'),mean_abs_selected_coef=('coef_abs','mean'),n_positive=('coef_standardized',lambda q:int((q>0).sum())),n_negative=('coef_standardized',lambda q:int((q<0).sum()))).sort_values('n_cell_types',ascending=False);save(tg,'models/TF_cross_cell_summary.csv',True)
sns.set_theme(style='whitegrid',font_scale=.8)
ff,axes=plt.subplots(2,3,figsize=(15,12))
for ax,c in zip(axes.flat,focus):
 d=tf[tf.cell_type==c].head(10).iloc[::-1];ax.barh(d.perturbation,d.coef_standardized,color=np.where(d.coef_standardized>0,'#b64b53','#3a7fa0'));ax.set_title(short[c]);ax.set_xlabel('Standardised Lasso coefficient');ax.axvline(0,c='black',lw=.6)
plt.tight_layout();fig('08_top_TFs')
plt.figure(figsize=(13,11));sns.heatmap(jac.rename(index=short,columns=short),cmap='viridis',vmin=0,vmax=1);plt.title('Lambert TFs: top-10 selected set Jaccard');fig('09_TF_overlap')
plt.figure(figsize=(14,12));sns.heatmap(cm.reindex(tg.head(30).index).rename(columns=short),cmap='vlag',center=0);plt.title('Lambert TF coefficients across targets');fig('10_TF_coefficients')
reps=[];ff,axes=plt.subplots(2,3,figsize=(14,9))
for ax,c in zip(axes.flat,focus):
 d=tf[tf.cell_type==c];pool=d[(d.coef_standardized>0)&(d.train_signature_delta_Pearson>0)&(d.selection_frequency_5_subsamples>=.6)]
 if pool.empty:pool=d[d.coef_standardized>0]
 if pool.empty:pool=d
 if pool.empty:continue
 r=pool.iloc[0];xx=x.loc[te,r.perturbation].to_numpy();yy=(b.loc[te,c]-b.loc[te,'hematopoietic stem cell']).to_numpy();rtest=pearsonr(xx,yy).statistic
 reps.append({'cell_type':c,'TF':r.perturbation,'coef_standardized':r.coef_standardized,'selection_frequency':r.selection_frequency_5_subsamples,'train_marginal_Pearson':r.train_signature_delta_Pearson,'test_marginal_Pearson':rtest})
 ax.hexbin(xx,yy,gridsize=30,mincnt=1,cmap='viridis');ax.set_title(f'{short[c]}: {r.perturbation}, test r={rtest:.2f}');ax.set_xlabel('MCF7 overexpression signature');ax.set_ylabel('Target − HSC')
plt.tight_layout();fig('11_representative_TF_match');reps=pd.DataFrame(reps);save(reps,'models/representative_TFs.csv')
# Predeclared lineage markers supplement the variance-based (ribosome-heavy) heatmap.
markers=['HBB','HBA1','HBA2','ALAS2','PPBP','PF4','GP9','LYZ','S100A8','S100A9','FCGR3A','MS4A1','CD79A','CD74','CD3D','CD3E','IL7R','NKG7','GNLY','GZMB','PRF1','CD34','KIT','GATA2']
markers=[v for v in markers if v in b.index and b.loc[v].std()>0]
zz=b.loc[markers].sub(b.loc[markers].mean(axis=1),axis=0).div(b.loc[markers].std(axis=1),axis=0);save(zz,'eda/lineage_marker_zscores.csv',True)
g=sns.clustermap(zz.rename(columns=short),row_cluster=False,col_cluster=True,method='ward',cmap='vlag',center=0,vmin=-2.5,vmax=2.5,figsize=(14,9),xticklabels=True,yticklabels=True)
for ext in ['png','pdf']:g.savefig(R/'figures'/f'12_lineage_marker_heatmap.{ext}',dpi=170)
plt.close(g.fig)
meta=json.loads((R/'metadata.json').read_text());meta['n_TF_features']=int(annotation.is_TF.sum());meta['TF_source']='https://raw.githubusercontent.com/phbradley/conga/173a414606cd5935298393daf4411fb2b13fed8d/conga/data/Lambert_et_al_PMID_29425488_TF_names_v_1.01.txt';meta['TF_annotation_note']='Lambert et al. Cell 2018 PMID 29425488, 1639 TFs v1.01, frozen mirror, exact symbols';meta['TF_list_sha256']=hashlib.sha256((R/'human_tfs.txt').read_bytes()).hexdigest();meta['verification']='all 21 heldout prediction vectors independently reconstructed; R2 rechecked; all convergence checks passed';meta['solver']={'selection':'random','random_state':42,'tol':1e-4,'max_iter':30000}
for mod in ['h5py','matplotlib','seaborn','threadpoolctl']:meta['versions'][mod]=importlib.metadata.version(mod)
(R/'metadata.json').write_text(json.dumps(meta,indent=2))
(R/'requirements.txt').write_text('\n'.join(f'{"scikit-learn" if k=="sklearn" else k}=={v}' for k,v in meta['versions'].items())+'\nreportlab\n')

qc=pd.read_csv(R/'qc/dataset_summary.csv');cv=pd.read_csv(R/'models/inner_CV_path.csv');ss=pd.read_csv(R/'models/log1p_sensitivity.csv');null=pd.read_csv(R/'models/shuffled_response_control.csv');clusters=pd.read_csv(R/'eda/cell_clusters_k5.csv')
boundary=sum(cv.loc[cv.groupby('cell_type').mean_validation_MSE.idxmin(),'alpha_ratio']<=.01000001)
doc=[]
def add(s):doc.append(s)
add('# 血液细胞与 L1000 扰动谱的 Lasso 分析报告\n\n日期：2026-10-26（Asia/Shanghai）。四部分分析均已完成；原始数据未修改。')
add('## 一、质控与预处理\n\n'+table(qc[['dataset','rows','columns','unique_gene_symbols','duplicate_rows_removed','missing','infinite','minimum','maximum']].rename(columns={'dataset':'数据','rows':'原行数','columns':'列数','unique_gene_symbols':'唯一基因数','duplicate_rows_removed':'重复行数','missing':'缺失','infinite':'无穷值','minimum':'最小值','maximum':'最大值'})))
add('血液数据删除 7 个重复行（涉及 MATR3、MKKS、HERC3、POLR2J3、NPIPA9、PINX1），L1000 删除 1 个重复行（MIA2）；均保留原文件第一次出现。两套数据均无缺失值和无穷值。血液基因包含 580 个 Ensembl 格式标识，L1000 为符号格式；这些 Ensembl 行未精确匹配到 L1000，本次未推断其符号映射。该损失已记录，后续可在核实版本后映射。')
add(f'按共同标识和相同顺序对齐后，共有 {len(split):,} 个基因、{len(names):,} 个扰动，其中 {meta["n_TF_features"]} 个扰动匹配 Lambert 人类 TF 名单。没有零方差扰动被剔除。血液表达范围 0–8.778474；其完整归一化来源未知，主分析采用提供尺度，不再盲目取对数。L1000 范围约 −10–10，作为有符号扰动特征使用，不取对数，也不将其声称为已核实的原始 log-fold change。')
add('髓系树突细胞约 94.0% 的基因表达为零，保留分析但标注为低信息目标。零值不作为缺失值填补。缺少原始计数与细胞/供体重复，因此不重新进行计数归一化，也不做不适用的差异表达显著性检验。\n\n![表达分布](figures/01_expression_distributions.png)\n\n![零比例](figures/02_zero_fraction.png)')
add('## 二、血液细胞探索性分析\n\n高变基因热图选取方差最大的 60 个基因，逐基因跨细胞 z 标准化，Ward/欧氏距离双向聚类。高变基因中核糖体基因较多，反映整体表达差异，不能仅据此解释谱系；另补充预先指定的血液谱系标记热图。PCA 和细胞聚类使用 2,000 个高变基因，5 类仅用于探索，不声称是最优分类数。')
add('![高变基因热图](figures/03_gene_heatmap.png)\n\n![谱系标记](figures/12_lineage_marker_heatmap.png)\n\n![PCA](figures/05_cell_PCA.png)')
for k,d in clusters.groupby('cluster_k5'):add(f'聚类 {k}：'+ '；'.join(d.cell_type.map(lambda q:short.get(q,q)))+'。')
add(f'红细胞/血小板、T/NK、单核/髓系细胞呈现谱系相关的聚集；HSC、祖细胞及若干 B/树突细胞则在当前表示下聚到同一组，提示这不是纯粹的谱系树。髓系树突细胞单独成簇可能同时受高零比例影响。PCA 前两轴解释比例为 {meta["pca_variance_ratio"][0]:.1%} 和 {meta["pca_variance_ratio"][1]:.1%}。全部相关性见 eda/cell_spearman_correlations.csv。')
add('## 三、Lasso 建模与预测评价\n\n对其余 21 种细胞分别构建 y = 目标细胞表达 − HSC 表达，X 为共同基因 × 全部 L1000 扰动。目标函数为 (1/2n)||y−截距−Xβ||² + α||β||₁。模型使用训练集内拟合的扰动列标准化与响应标准化，系数表保留符号、标准化系数及原始单位系数。')
add(f'共同基因固定随机划分为 {len(tr):,} 个训练基因和 {len(te):,} 个测试基因。训练集内 3 折交叉验证搜索 24 个相对 alpha，采用 1 标准误规则偏向稀疏模型。测试集不调参，候选来自训练集模型，未用测试基因重新拟合。最终数值收敛检查通过，所有拟合最多使用 {int(dg.max_iterations.max())} 次迭代。该矩阵 p<n，属于多特征稀疏回归，非严格的 p>n。')
summary=m[['cell_type','test_R2','test_RMSE','test_Pearson','n_nonzero','n_TF_nonzero']].copy();summary.cell_type=summary.cell_type.map(short);summary.columns=['目标','测试R²','测试RMSE','测试Pearson','非零扰动数','非零TF数'];add(table(summary))
add('![测试R2](figures/06_test_R2.png)\n\n![测试预测](figures/07_heldout_predictions.png)')
add(f'全部模型优于训练均值基线，测试 R² 范围 {m.test_R2.min():.3f}–{m.test_R2.max():.3f}；红细胞和中性粒细胞较高，共同髓系祖细胞较低。六个代表目标的单次训练响应打乱对照测试 R² 均为负（{null.shuffled_train_response_test_R2.min():.3f} 至 {null.shuffled_train_response_test_R2.max():.3f}），仅说明打乱后失去匹配能力，不是正式显著性检验。')
add(f'CV 路径中有 {boundary} 个目标的最小验证 MSE 位于最弱惩罚边界；已明确记录，后续可扩大搜索范围。主模型均按事先规定的 1SE 规则选取，未根据测试表现挑选 alpha。')
add('## 四、候选扰动、TF 与跨细胞比较\n\n全部扰动按 |标准化系数| 排序；只有匹配 Lambert 1,639 人类 TF 名单的候选进入 TF 表。RIOK2、DNMT3A、HDAC1、HMGB1、PML 等可作为其他调控扰动讨论，但不在本报告中误称为严格 TF。正系数表示联合拟合中的同向加权，负系数表示反向加权，不等价于已验证的过表达/敲除实验效果。')
add('以下为六个代表目标各自按绝对系数排名前五的 TF，括号为带符号标准化系数：')
for c in focus:
 d=tf[tf.cell_type==c].head(5);add(cn[c]+'：'+'，'.join(f'{q.perturbation} ({q.coef_standardized:+.3f})' for q in d.itertuples())+'。')
add('![候选TF](figures/08_top_TFs.png)\n\n![TF系数比较](figures/10_TF_coefficients.png)')
rep=reps.copy();rep.cell_type=rep.cell_type.map(cn);rep.columns=['目标','代表TF','标准化系数','5次入选频率','训练边际相关','测试边际相关'];add(table(rep))
add('代表 TF 优先选择正系数、正训练边际相关且入选频率至少 60% 的最强候选；不使用测试相关选择候选。上表和图中，单个 TF 的测试相关通常明显低于联合模型的预测相关，说明这些候选只覆盖差异的一部分。五次入选不等于高置信因果证据，也不构成稳定性选择的错误率保证。')
add('![代表TF匹配](figures/11_representative_TF_match.png)')
add('跨细胞比较使用每个目标前 10 个非零 TF 的 Jaccard：')
best=ov.sort_values('Jaccard',ascending=False).head(8).copy();best.cell_a=best.cell_a.map(short);best.cell_b=best.cell_b.map(short);add(table(best[['cell_a','cell_b','n_shared','Jaccard','shared_TFs']]))
add('![TF重叠](figures/09_TF_overlap.png)\n\n谱系相近目标通常共享更多候选，尤其单核细胞及其亚型。也存在跨谱系共享，可能来自普遍转录程序或共同的 HSC 参照，不应直接解释为谱系特异因果调控。集合重叠不考虑符号，需结合带符号的系数矩阵查看同向或反向。')
add('## 稳定性、预处理敏感性与解释边界')
add('扰动库的覆盖范围也是关键限制：GATA2、MYB 有对应扰动列，但 GATA1、SPI1、CEBPA、RUNX1、TAL1、KLF1、PAX5、BCL11A、TCF7 在此文件中没有对应扰动列。它们作为响应基因出现不等于存在相应干预。模型无法选出未被提供的扰动，不能把这些经典造血 TF 未入选解释为其不重要。详细覆盖表见 qc/canonical_TF_coverage.csv。')
add(f'五次训练基因 80% 子抽样已完成，全部候选的入选和同符号频率已保存。额外 log1p 后，入选扰动集合 Jaccard 范围为 {ss.selected_set_Jaccard.min():.3f}–{ss.selected_set_Jaccard.max():.3f}，前 20 扰动重叠为 {int(ss.top20_overlap_count.min())}–{int(ss.top20_overlap_count.max())} 个，说明预处理会改变部分候选，不能把单一排序视为唯一结果。此敏感性模型复用相对 alpha、未重新调参，R² 对应不同响应尺度，不用于宣称取对数后的性能必然更优。')
add('本分析是 MCF7 扰动特征与血液细胞平均表达差异的统计匹配，不证明 MCF7 中的扰动在血液背景中同样有效。PAX8、OVOL2 等候选即使排名较高，也需警惕组织背景或共线性的影响，不能直接称为血液分化主调控因子。优先核实候选 TF 在目标细胞的表达、相关文献及独立血液扰动证据，再考虑过表达、敲低及组合干预。')
add('基因不是独立生物学重复；随机留出基因的 R² 仅评价当前平均谱上的差异重建，不能外推为新供体预测准确率、分化成功率或实验因果效果。基因间共表达可能让随机划分偏乐观。建议用独立供体数据、按共表达模块/通路划分基因的验证，以及血液细胞背景的扰动实验检验候选。现有数据不能估计供体间变异或计算适当的差异表达 p 值。')
add('## 文件与方法来源\n\n完整质控见 qc/；探索数据见 eda/；全候选、TF 排名、测试指标、逐基因预测、稳定性和敏感性见 models/；图像为 figures/ 下的 PNG/PDF。README.md 记录全部参数，metadata.json 保存输入校验和与依赖版本。\n\nTF 来源：Lambert SA et al. The Human Transcription Factors. Cell. 2018;172:650–665. doi:10.1016/j.cell.2018.01.029；PMID 29425488。名单 v1.01 的固定 GitHub 镜像及 SHA256 见 metadata.json。没有将示例 notebook 中训练集分数当作交叉验证分数。')
(R/'REPORT.md').write_text('\n\n'.join(doc)+'\n')
# PDF uses built-in CJK CID font; all analysis plots are embedded.
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Image,Table,TableStyle,PageBreak
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
pdfmetrics.registerFont(UnicodeCIDFont('STSong-Light'))
styles=getSampleStyleSheet()
for key in ['Normal','Heading1','Heading2','Heading3']:
 styles[key].fontName='STSong-Light';styles[key].wordWrap='CJK';styles[key].leading=16 if key=='Normal' else 21
styles['Normal'].fontSize=10
story=[]
for block in '\n\n'.join(doc).split('\n\n'):
 if not block.strip():continue
 if block.startswith('!['):
  p=re.search(r'\]\((.*?)\)',block)
  if p:
   im=Image(str(R/p.group(1)));w,h=im.imageWidth,im.imageHeight;scale=min(480/w,510/h);im.drawWidth=w*scale;im.drawHeight=h*scale;story.extend([im,Spacer(1,12)])
 elif block.startswith('|'):
  lines=[v for v in block.splitlines() if not re.match(r'^\|\s*---',v)]
  rows=[[Paragraph(html.escape(c.strip()),ParagraphStyle('small',parent=styles['Normal'],fontSize=7,leading=10)) for c in v.strip('|').split('|')] for v in lines]
  tb=Table(rows,colWidths=[480/len(rows[0])]*len(rows[0]),repeatRows=1,hAlign='LEFT');tb.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e7eef4')),('GRID',(0,0),(-1,-1),.3,colors.lightgrey),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),3),('RIGHTPADDING',(0,0),(-1,-1),3)]));story.extend([tb,Spacer(1,12)])
 else:
  level=len(block)-len(block.lstrip('#'));text=html.escape(block.lstrip('# ')).replace('−','-').replace('β','beta').replace('α','alpha').replace('₁','1').replace('²','^2').replace('\n','<br/>');style=styles['Heading1' if level==1 else 'Heading2' if level else 'Normal'];story.extend([Paragraph(text,style),Spacer(1,10)])
def footer(canvas,doc):canvas.setFont('STSong-Light',9);canvas.drawRightString(540,25,str(doc.page))
SimpleDocTemplate(str(R/'REPORT.pdf'),pagesize=(595,842),leftMargin=55,rightMargin=55,topMargin=45,bottomMargin=40,title='血液细胞与L1000扰动谱的Lasso分析',author='Codex').build(story,onFirstPage=footer,onLaterPages=footer)
print('Verified all 21 models; TF features:',meta['n_TF_features']);print(reps.to_string(index=False));print('Report saved:',R/'REPORT.md')
