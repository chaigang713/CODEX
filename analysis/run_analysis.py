"""Run with python run_analysis.py --data /path/to/CODEX; outputs next to script."""
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import h5py, scipy, sklearn
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.stats import pearsonr, spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import lasso_path
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_squared_error
from threadpoolctl import threadpool_limits

ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True);args=ap.parse_args()
ROOT=Path(__file__).resolve().parent
for folder in ['qc','eda','models','figures']: (ROOT/folder).mkdir(exist_ok=True)
def save(df,path,index=True):df.to_csv(ROOT/path,index=index)
def fig(name):
    for ext in ['png','pdf']:plt.savefig(ROOT/'figures'/f'{name}.{ext}',dpi=170,bbox_inches='tight')
    plt.close()
def sha(p):
    with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
HSC='hematopoietic stem cell'
FOCUS=['erythrocyte','platelet','monocyte','B cell','CD4-positive, alpha-beta T cell','natural killer cell']
short=dict(zip(['hematopoietic stem cell','common myeloid progenitor','erythrocyte','platelet','macrophage','B cell','monocyte','natural killer cell','CD4-positive, alpha-beta T cell','CD8-positive, alpha-beta T cell','basophil','neutrophil','myeloid dendritic cell','plasmacytoid dendritic cell','plasma cell','mature NK T cell','regulatory T cell','classical monocyte','non-classical monocyte','naive thymus-derived CD4-positive, alpha-beta T cell','intermediate monocyte','hematopoietic precursor cell'],['HSC','CMP','Erythrocyte','Platelet','Macrophage','B cell','Monocyte','NK cell','CD4 T','CD8 T','Basophil','Neutrophil','Myeloid DC','Plasmacytoid DC','Plasma cell','NKT','Treg','Classical mono','Nonclassical mono','Naive CD4 T','Intermediate mono','Precursor']))
sns.set_theme(style='whitegrid',font_scale=.8)
start=time.time();bp=args.data/'blood mean expr.csv';xp=args.data/'L1000 MCF7_OE gene expression.h5'
b=pd.read_csv(bp,index_col=0)
with h5py.File(xp) as f:x=pd.DataFrame(f['expression/block0_values'][:],index=[z.decode() for z in f['expression/axis1'][:]],columns=[z.decode() for z in f['expression/axis0'][:]])
audit=[]
for label,d in [('blood',b),('L1000',x)]:
    old=d.index.astype(str);d.index=old.str.strip();v=d.to_numpy(dtype=float)
    save(pd.DataFrame({'original':old,'normalized':d.index}),f'qc/{label}_gene_identifiers.csv',False)
    save(d[d.index.duplicated(keep=False)],f'qc/{label}_duplicate_genes.csv')
    audit.append({'dataset':label,'rows':d.shape[0],'columns':d.shape[1],'unique_gene_symbols':d.index.nunique(),'duplicate_rows_removed':int(d.index.duplicated().sum()),'missing':int(np.isnan(v).sum()),'infinite':int(np.isinf(v).sum()),'negative_entries':int((v<0).sum()),'zero_fraction':float((v==0).mean()),'minimum':float(np.nanmin(v)),'maximum':float(np.nanmax(v)),'empty_gene_names':int((d.index=='').sum()),'ensembl_style_ids':int(d.index.str.match(r'^ENSG\d+').sum()),'whitespace_names_changed':int((old!=d.index).sum())})
    assert np.isfinite(v).all()
save(pd.DataFrame(audit),'qc/dataset_summary.csv',False)
b=b.loc[~b.index.duplicated(keep='first')];x=x.loc[~x.index.duplicated(keep='first')]
assert b.columns.is_unique and x.columns.is_unique and HSC in b
cq=b.describe(percentiles=[.25,.5,.75,.95,.99]).T;cq['zero_fraction']=(b==0).mean()
save(cq,'qc/blood_cell_summary.csv')
common=b.index.intersection(x.index,sort=False)
save(pd.DataFrame({'gene_symbol':common}),'qc/shared_genes.csv',False)
for name,a in [('blood_only',b.index.difference(x.index)),('L1000_only',x.index.difference(b.index))]:save(pd.DataFrame({'gene_symbol':a}),f'qc/{name}_genes.csv',False)
tfs=set((ROOT/'human_tfs.txt').read_text().split())
save(pd.DataFrame({'perturbation':x.columns,'is_TF':x.columns.isin(tfs)}),'qc/perturbation_TF_annotation.csv',False)
print('QC',len(common),'shared genes;',sum(x.columns.isin(tfs)),'TF perturbations',flush=True)
# EDA: gene-wise z scoring is restricted to visualisation and PCA.
plt.figure(figsize=(12,5));sns.boxplot(data=b.rename(columns=short),showfliers=False,color='#91b9ce');plt.xticks(rotation=65,ha='right');plt.ylabel('Expression, provided scale');fig('01_expression_distributions')
plt.figure(figsize=(11,4));cq.zero_fraction.rename(index=short).plot.bar(color='#497d9b');plt.ylabel('Fraction of zero genes');fig('02_zero_fraction')
var=b.var(axis=1);top=var.nlargest(60).index
z=b.loc[top].sub(b.loc[top].mean(axis=1),axis=0).div(b.loc[top].std(axis=1),axis=0)
save(z,'eda/top60_gene_zscores.csv')
g=sns.clustermap(z.rename(columns=short),method='ward',metric='euclidean',cmap='vlag',center=0,vmin=-2.5,vmax=2.5,figsize=(15,15),xticklabels=True,yticklabels=True)
g.fig.suptitle('60 highest-variance genes; row z-scores',y=1.01)
for ext in ['png','pdf']:g.savefig(ROOT/'figures'/f'03_gene_heatmap.{ext}',dpi=170)
plt.close(g.fig)
save(b.corr(method='spearman'),'eda/cell_spearman_correlations.csv')
g=sns.clustermap(b.corr(method='spearman').rename(index=short,columns=short),cmap='viridis',figsize=(11,11),vmin=0,vmax=1)
for ext in ['png','pdf']:g.savefig(ROOT/'figures'/f'04_cell_correlations.{ext}',dpi=170)
plt.close(g.fig)
vt=var.nlargest(2000).index;zz=b.loc[vt].sub(b.loc[vt].mean(axis=1),axis=0).div(b.loc[vt].std(axis=1),axis=0).T
lk=linkage(zz,method='ward',metric='euclidean')
save(pd.DataFrame({'cell_type':zz.index,'cluster_k5':fcluster(lk,5,criterion='maxclust')}),'eda/cell_clusters_k5.csv',False)
save(pd.DataFrame(lk,columns=['left','right','distance','n_cells']),'eda/cell_linkage.csv',False)
pca=PCA(n_components=2);pc=pca.fit_transform(zz)
save(pd.DataFrame(pc,index=zz.index,columns=['PC1','PC2']),'eda/cell_PCA.csv')
plt.figure(figsize=(12,8));plt.scatter(pc[:,0],pc[:,1],s=65,c=['#c75050' if c==HSC else '#3b819c' for c in zz.index])
for c,(a,bb) in zip(zz.index,pc):plt.annotate(short[c],(a,bb),fontsize=8,xytext=(4,3),textcoords='offset points')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})');plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})');fig('05_cell_PCA')

cells=[c for c in b if c!=HSC];X=x.loc[common].to_numpy(dtype=np.float64)
Y=b.loc[common,cells].sub(b.loc[common,HSC],axis=0).to_numpy(dtype=np.float64)
vary=X.std(axis=0)>1e-10;names=x.columns[vary];X=X[:,vary]
idx=np.arange(len(common));train,test=train_test_split(idx,test_size=.2,random_state=42)
save(pd.DataFrame({'gene_symbol':common,'split':np.where(np.isin(idx,test),'test','train')}),'models/gene_split.csv',False)
ratios=np.geomspace(1,.01,24);folds=list(KFold(3,shuffle=True,random_state=43).split(train))
def prepare(xx,yy):
    sc=StandardScaler();xx=np.asfortranarray(sc.fit_transform(xx));ym=yy.mean(axis=0);ys=yy.std(axis=0);ys=np.where(ys>1e-12,ys,1.)
    yy=(yy-ym)/ys;return xx,yy,sc,ym,ys,np.ascontiguousarray(xx.T@xx),np.ascontiguousarray(xx.T@yy)
diagnostics=[]
def pathfit(xx,yy,gram,xy,alphas,label):
    al,co,dual,nit=lasso_path(xx,np.ascontiguousarray(yy),alphas=alphas,precompute=gram,Xy=np.ascontiguousarray(xy),max_iter=30000,tol=1e-4,selection='random',random_state=42,return_n_iter=True)
    diagnostics.append({'fit':label,'max_iterations':int(max(nit)),'max_dual_gap':float(np.max(dual)),'n_solutions':len(al)})
    if max(nit)>=30000:raise RuntimeError('Lasso convergence failure '+label)
    return co
metrics=[];coefall=[];tfall=[];predall=[];cvrows=[];reps=[];sens=[];cache=[]
with threadpool_limits(limits=2):
    for fi,(a,v) in enumerate(folds):
        pp=prepare(X[train[a]],Y[train[a]]);cache.append((pp,pp[2].transform(X[train[v]]),Y[train[v]],fi))
    xx,yy,sc,ym,ys,gram,xy=prepare(X[train],Y[train]);xtest=sc.transform(X[test]);chosen=[];primary=[]
    for j,c in enumerate(cells):
        errs=[]
        for (fx,fy,fs,fm,fd,fg,fxy),vx,vy,fi in cache:
            am=max(np.max(np.abs(fxy[:,j]))/len(fx),1e-12);co=pathfit(fx,fy[:,j],fg,fxy[:,j],am*ratios,f'cv{fi}/{c}')
            pp=vx@co*fd[j]+fm[j];errs.append(np.mean((pp-vy[:,j,None])**2,axis=0))
        errs=np.array(errs);mean=errs.mean(axis=0);se=errs.std(axis=0,ddof=1)/np.sqrt(3);best=mean.argmin();pick=np.where(mean<=mean[best]+se[best])[0][0]
        ratio=ratios[pick];chosen.append(ratio);alpha=max(np.max(np.abs(xy[:,j]))/len(xx),1e-12)*ratio
        cf=pathfit(xx,yy[:,j],gram,xy[:,j],[alpha],f'train/{c}')[:,0];primary.append(cf)
        pred=xtest@cf*ys[j]+ym[j];actual=Y[test,j];trpred=xx@cf*ys[j]+ym[j];nz=np.abs(cf)>1e-9
        row={'cell_type':c,'alpha_ratio':ratio,'alpha_standardized_y':alpha,'n_genes_train':len(train),'n_genes_test':len(test),'n_nonzero':int(nz.sum()),'n_TF_nonzero':int((nz&names.isin(tfs)).sum()),'test_R2':r2_score(actual,pred),'test_RMSE':np.sqrt(mean_squared_error(actual,pred)),'test_Pearson':pearsonr(actual,pred).statistic if pred.std()>1e-12 else np.nan,'test_Spearman':spearmanr(actual,pred).statistic if pred.std()>1e-12 else np.nan,'baseline_R2':r2_score(actual,np.repeat(ym[j],len(test))),'baseline_RMSE':np.sqrt(np.mean((actual-ym[j])**2)),'train_R2':r2_score(Y[train,j],trpred),'blood_zero_fraction':cq.loc[c,'zero_fraction']};metrics.append(row)
        for k in range(len(ratios)):cvrows.append({'cell_type':c,'alpha_ratio':ratios[k],'mean_validation_MSE':mean[k],'SE_validation_MSE':se[k],'chosen_1SE':k==pick})
        predall.append(pd.DataFrame({'gene_symbol':common[test],'cell_type':c,'observed_delta':actual,'predicted_delta':pred,'baseline_delta':ym[j]}))
        print(f'Model {j+1}/21 {c}: test R2={row["test_R2"]:.3f}, selected={nz.sum()}',flush=True)
    # Stability: five 80% subsamples of training genes, fixed chosen alpha ratio.
    primary=np.array(primary).T;freq=np.zeros_like(primary);agree=np.zeros_like(primary)
    for rep in range(5):
        sub=np.random.default_rng(100+rep).choice(train,size=int(.8*len(train)),replace=False);sx,sy,ss,sm,sd,sg,sxy=prepare(X[sub],Y[sub])
        for j,c in enumerate(cells):
            alpha=max(np.max(np.abs(sxy[:,j]))/len(sx),1e-12)*chosen[j];cc=pathfit(sx,sy[:,j],sg,sxy[:,j],[alpha],f'stability{rep}/{c}')[:,0]
            freq[:,j]+=np.abs(cc)>1e-9;agree[:,j]+=(np.abs(cc)>1e-9)&(np.sign(cc)==np.sign(primary[:,j]))
        print('Stability repeat',rep+1,'complete',flush=True)
    freq/=5;agree/=5
    # Sensitivity to an additional log1p: same split and alpha ratio, no retuning.
    LY=np.log1p(b.loc[common,cells]).sub(np.log1p(b.loc[common,HSC]),axis=0).to_numpy()
    for j,c in enumerate(cells):
        ly=LY[train,j];lm=ly.mean();ld=ly.std();lz=(ly-lm)/ld;lxy=xx.T@lz;alpha=max(np.max(np.abs(lxy))/len(xx),1e-12)*chosen[j]
        cc=pathfit(xx,lz,gram,lxy,[alpha],f'log1p/{c}')[:,0];pp=xtest@cc*ld+lm
        old=set(names[np.abs(primary[:,j])>1e-9]);new=set(names[np.abs(cc)>1e-9]);ta=set(names[np.argsort(-np.abs(primary[:,j]))[:20]])&old;tb=set(names[np.argsort(-np.abs(cc))[:20]])&new
        sens.append({'cell_type':c,'log1p_test_R2':r2_score(LY[test,j],pp),'selected_set_Jaccard':len(old&new)/max(len(old|new),1),'top20_overlap_count':len(ta&tb),'coefficient_Spearman':spearmanr(primary[:,j],cc).statistic,'note':'same alpha ratio, not retuned; responses use different scales'})
    for j,c in enumerate(cells):
        cor=xx.T@yy[:,j]/len(xx)
        d=pd.DataFrame({'cell_type':c,'perturbation':names,'is_TF':names.isin(tfs),'coef_standardized':primary[:,j],'coef_abs':np.abs(primary[:,j]),'coef_expression_per_X_SD':primary[:,j]*ys[j],'coef_original_X_scale':primary[:,j]*ys[j]/sc.scale_,'selection_frequency_5_subsamples':freq[:,j],'same_sign_frequency_5_subsamples':agree[:,j],'train_signature_delta_Pearson':cor})
        d['selected']=d.coef_abs>1e-9;d['direction']=np.where(d.coef_standardized>1e-9,'positive',np.where(d.coef_standardized< -1e-9,'negative','zero'));d=d.sort_values('coef_abs',ascending=False);d['rank_abs_coefficient']=np.arange(1,len(d)+1)
        coefall.append(d);tfall.append(d[d.is_TF&d.selected])
    null=[]
    for j,c in enumerate(cells):
        if c not in FOCUS:continue
        yp=np.random.default_rng(2026+j).permutation(Y[train,j]);pm=yp.mean();pdv=yp.std();yp=(yp-pm)/pdv;pxy=xx.T@yp;alpha=max(np.max(np.abs(pxy))/len(xx),1e-12)*chosen[j]
        cc=pathfit(xx,yp,gram,pxy,[alpha],f'null/{c}')[:,0];null.append({'cell_type':c,'shuffled_train_response_test_R2':r2_score(Y[test,j],xtest@cc*pdv+pm)})
    save(pd.DataFrame(null),'models/shuffled_response_control.csv',False)
m=pd.DataFrame(metrics);save(m,'models/test_metrics.csv',False)
ac=pd.concat(coefall,ignore_index=True);tf=pd.concat(tfall,ignore_index=True)
save(ac,'models/all_perturbation_coefficients.csv',False);save(tf,'models/selected_TFs.csv',False)
save(ac[ac.selected].groupby('cell_type',sort=False).head(20),'models/top20_perturbations.csv',False);save(tf.groupby('cell_type',sort=False).head(10),'models/top10_TFs.csv',False)
save(pd.concat(predall),'models/heldout_gene_predictions.csv.gz',False);save(pd.DataFrame(cvrows),'models/inner_CV_path.csv',False)
save(pd.DataFrame(sens),'models/log1p_sensitivity.csv',False);save(pd.DataFrame(diagnostics),'models/convergence_diagnostics.csv',False)
cm=pd.DataFrame(primary,index=names,columns=cells);save(cm,'models/coefficient_matrix_standardized.csv')
save(pd.DataFrame({'perturbation':names,'training_mean':sc.mean_,'training_SD':sc.scale_}),'models/feature_scaler.csv',False)
original_coefficients=primary*ys[None,:]/sc.scale_[:,None]
save(pd.DataFrame({'cell_type':cells,'response_training_mean':ym,'response_training_SD':ys,'intercept_original_X':ym-sc.mean_@original_coefficients}),'models/model_parameters.csv',False)
sets={c:set(tf[tf.cell_type==c].head(10).perturbation) for c in cells}
jac=pd.DataFrame([[len(sets[a]&sets[bb])/max(len(sets[a]|sets[bb]),1) for bb in cells] for a in cells],index=cells,columns=cells);save(jac,'models/top10_TF_Jaccard.csv')
ov=[]
for i,a in enumerate(cells):
 for bb in cells[i+1:]:ov.append({'cell_a':a,'cell_b':bb,'n_shared':len(sets[a]&sets[bb]),'Jaccard':jac.loc[a,bb],'shared_TFs':';'.join(sorted(sets[a]&sets[bb]))})
save(pd.DataFrame(ov),'models/TF_pairwise_overlap.csv',False)
tg=tf.groupby('perturbation').agg(n_cell_types=('cell_type','nunique'),mean_abs_selected_coef=('coef_abs','mean'),n_positive=('coef_standardized',lambda q:int((q>0).sum())),n_negative=('coef_standardized',lambda q:int((q<0).sum()))).sort_values('n_cell_types',ascending=False);save(tg,'models/TF_cross_cell_summary.csv')
plt.figure(figsize=(10,8));mm=m.sort_values('test_R2');plt.barh(mm.cell_type.map(short),mm.test_R2,color=['#b54d4d' if q>.9 else '#417f9c' for q in mm.blood_zero_fraction]);plt.axvline(0,c='black',lw=.8);plt.xlabel('Held-out gene R²');fig('06_test_R2')
figu,axes=plt.subplots(2,3,figsize=(14,9));pr=pd.concat(predall)
for ax,c in zip(axes.flat,FOCUS):
    d=pr[pr.cell_type==c];ax.hexbin(d.observed_delta,d.predicted_delta,gridsize=35,mincnt=1,cmap='viridis');lim=[min(d.observed_delta.min(),d.predicted_delta.min()),max(d.observed_delta.max(),d.predicted_delta.max())];ax.plot(lim,lim,'r--',lw=1);ax.set_title(f'{short[c]} | R²={m.set_index("cell_type").loc[c,"test_R2"]:.3f}');ax.set_xlabel('Observed target − HSC');ax.set_ylabel('Predicted target − HSC')
plt.tight_layout();fig('07_heldout_predictions')
figu,axes=plt.subplots(2,3,figsize=(15,12))
for ax,c in zip(axes.flat,FOCUS):
    d=tf[tf.cell_type==c].head(10).iloc[::-1];ax.barh(d.perturbation,d.coef_standardized,color=np.where(d.coef_standardized>0,'#b64b53','#3a7fa0'));ax.set_title(short[c]);ax.set_xlabel('Standardised Lasso coefficient');ax.axvline(0,c='black',lw=.6)
plt.tight_layout();fig('08_top_TFs')
plt.figure(figsize=(13,11));sns.heatmap(jac.rename(index=short,columns=short),cmap='viridis',vmin=0,vmax=1);plt.title('Top-10 selected TF Jaccard similarity');fig('09_TF_overlap')
plt.figure(figsize=(14,12));sns.heatmap(cm.reindex(tg.head(30).index).rename(columns=short),cmap='vlag',center=0);plt.title('TF coefficients across targets');fig('10_TF_coefficients')
figu,axes=plt.subplots(2,3,figsize=(14,9))
for ax,c in zip(axes.flat,FOCUS):
    d=tf[tf.cell_type==c];pool=d[(d.coef_standardized>0)&(d.train_signature_delta_Pearson>0)&(d.selection_frequency_5_subsamples>=.6)]
    if pool.empty:pool=d[d.coef_standardized>0]
    if pool.empty:pool=d
    if pool.empty:ax.set_title(short[c]+' (no selected TF)');continue
    rr=pool.iloc[0];k=list(names).index(rr.perturbation);j=cells.index(c);testcor=pearsonr(X[test,k],Y[test,j]).statistic
    reps.append({'cell_type':c,'TF':rr.perturbation,'coef_standardized':rr.coef_standardized,'selection_frequency':rr.selection_frequency_5_subsamples,'train_marginal_Pearson':rr.train_signature_delta_Pearson,'test_marginal_Pearson':testcor})
    ax.hexbin(X[test,k],Y[test,j],gridsize=30,mincnt=1,cmap='viridis');ax.set_title(f'{short[c]}: {rr.perturbation}, test r={testcor:.2f}');ax.set_xlabel('MCF7 overexpression signature');ax.set_ylabel('Target − HSC')
plt.tight_layout();fig('11_representative_TF_match');save(pd.DataFrame(reps),'models/representative_TFs.csv',False)
meta={'blood_sha256':sha(bp),'L1000_sha256':sha(xp),'TF_list_sha256':sha(ROOT/'human_tfs.txt'),'TF_source':'https://raw.githubusercontent.com/phbradley/conga/173a414606cd5935298393daf4411fb2b13fed8d/conga/data/Lambert_et_al_PMID_29425488_TF_names_v_1.01.txt','TF_annotation_note':'Lambert et al. Cell 2018 PMID 29425488; 1639 human TFs, v1.01; exact-symbol matching, no alias inference','n_common_genes':len(common),'n_features':len(names),'n_TF_features':int(names.isin(tfs).sum()),'pca_variance_ratio':pca.explained_variance_ratio_.tolist(),'test_seed':42,'inner_CV_seed':43,'inner_folds':3,'alpha_ratios':ratios.tolist(),'alpha_rule':'1SE, most regularised eligible model','n_stability_subsamples':5,'stability_fraction':.8,'response_scale':'provided blood scale; no extra log in primary analysis','versions':{'numpy':np.__version__,'pandas':pd.__version__,'sklearn':sklearn.__version__,'scipy':scipy.__version__,'h5py':h5py.__version__},'runtime_seconds':time.time()-start}
(ROOT/'metadata.json').write_text(json.dumps(meta,indent=2));print('ALL COMPLETE',meta['runtime_seconds'],flush=True)
