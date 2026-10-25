"""Build the two self-contained assignment deliverables from verified analysis results."""
from pathlib import Path
import re,html,zipfile,hashlib
import pandas as pd
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Image,Table,TableStyle,KeepTogether
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont

R=Path(__file__).resolve().parent;dest=R.parent/'submission';dest.mkdir(exist_ok=True)
base=(R/'REPORT.md').read_text()
base=base[base.index('## 一、质控与预处理'):]
base=base[:base.index('## 文件与方法来源')]
intro='''# 以造血干细胞为参考的血液细胞表达差异与 L1000 扰动谱的 Lasso 匹配分析

第二次作业｜分析报告｜2026 年 10 月 26 日

## 摘要

本研究以造血干细胞（HSC）为共同参考，利用课程提供的血液细胞平均表达矩阵与 MCF7 过表达扰动特征，检验多个 L1000 扰动的线性组合能否重建不同血液细胞相对 HSC 的表达差异，并筛选候选转录因子（TF）。去重及标识对齐后纳入 11,940 个基因与 2,717 个扰动，其中 286 个扰动匹配人类 TF 名单。对 21 个目标细胞分别拟合 Lasso，在训练集内调参，并在独立留出的 2,388 个基因上评价。测试 R² 为 0.071–0.621，其中红细胞为 0.621、中性粒细胞为 0.594。红细胞模型的候选 MYB 与测试基因差异的边际相关为 0.350；其他代表候选的单因子相关较弱，不能替代联合模型。单核细胞与经典单核细胞的前 10 个 TF 集合相同，也观察到跨谱系候选差异。结果支持以扰动表达特征提出可验证的调控假说，但受细胞背景、扰动库覆盖、汇总数据和验证设计限制，不能直接推断因果分化或重编程效率。

## 研究问题与数据

造血分化伴随多种转录调控程序变化。本研究的问题是：①哪些 MCF7 过表达扰动特征可以共同拟合目标血液细胞相对 HSC 的表达差异？②不同谱系是否共享候选 TF？③这种拟合在未用于训练的基因上表现如何，候选对基因重抽样和预处理是否敏感？这里的“匹配”是表达模式的统计匹配，而不是已经证明的细胞转化机制。

血液数据来自课程提供的 blood mean expr.csv，行为基因标识、列为 22 种细胞类型，HSC 列为 hematopoietic stem cell。L1000 数据来自课程提供的 L1000 MCF7_OE gene expression.h5，HDF5 属性明确其行为 gene_symbol、列为 cmap_name。数据从 chaigang713/CODEX 仓库读取，输入校验和已记录。CMap/L1000 的一般原理见文献 [1]；本文件的具体数据级别、处理时间、剂量及血液矩阵的原始细胞/供体组成未被课程文件完整说明，本研究不将平台文献的信息自动当作该文件的元数据。

参考细胞固定为 HSC，其余 21 个细胞全部建模；重点展示红细胞、血小板、单核细胞、B 细胞、CD4 T 细胞和 NK 细胞，覆盖不同谱系。细胞类型是平均谱，不存在可用于估计生物学重复变异的供体级观测。

'''
base=base.replace('只有匹配 Lambert 1,639 人类 TF 名单的候选进入 TF 表。','只有匹配 Lambert 等人整理的 1,639 人类 TF 名单 [2] 的候选进入 TF 表。')
base=base.replace('目标函数为 (1/2n)||y−截距−Xβ||² + α||β||₁。','采用 Lasso [3]，目标函数为 (1/2n)||y−截距−Xβ||² + α||β||₁。')
base=base.replace('采用 1 标准误规则偏向稀疏模型。','每个训练折独立计算 alpha_max = max|Xᵀy|/n，24 个 alpha/alpha_max 在 1 至 0.01 间按对数间隔取值；采用最小平均验证 MSE 加其 1 标准误范围内惩罚最大的模型（1SE 规则），偏向稀疏解。')
base=base.replace('详细覆盖表见 qc/canonical_TF_coverage.csv。','这些缺失扰动的覆盖核对已保存。')
base=base.replace('全部相关性见 eda/cell_spearman_correlations.csv。','相关性分析另使用全部去重血液基因的 Spearman 相关，作为聚类解释的补充。')
base=base.replace('## 稳定性、预处理敏感性与解释边界','## 五、稳健性与局限性')
base=base.replace('集合重叠不考虑符号，需结合带符号的系数矩阵查看同向或反向。','集合重叠不考虑符号，需结合带符号的系数矩阵查看同向或反向。差异方面，红细胞与血小板仅共享前10 TF中的KLF6（Jaccard=0.053）；血小板与B细胞无重叠（Jaccard=0）；CD4 T与NK共享4个（Jaccard=0.250）。因此，部分共享程序与明显的目标间差异同时存在。')
bio='''## 六、生物学解释：符合预期与待验证结果

### 6.1 谱系相关的聚集与 MYB 候选

T/NK 细胞、单核/髓系细胞在表达聚类中分别相近，单核细胞亚型也共享较多候选 TF，符合相近细胞状态包含共享表达程序的预期。然而红细胞和血小板聚集、HSC 与部分 B/树突细胞同组，不能简单视为真实发育树；平均表达、核糖体/代谢程序及低 RNA 特征都可能影响距离。预设标记热图中的 HBB/HBA、PPBP/PF4、LYZ/S100A8 等模式可作为细胞身份的辅助核查，不作为差异表达显著性证据。

MYB 在红细胞模型中系数为 +0.073，五次子抽样均入选，测试基因上的单扰动边际相关为 0.350，具有相对明确的同向匹配。小鼠 c-myb 缺失损害胎肝造血和成体型红细胞生成 [4]，因此 MYB 与红系/造血程序相关这一点有实验先验。但 MYB 的作用具有发育阶段和剂量依赖，且本研究的特征来自 MCF7；正系数不能推出“持续过表达 MYB 即可使 HSC 成熟为红细胞”。本结果支持优先验证表达程序重叠，而不直接指定干预方案。

### 6.2 KLF6：髓系先验与跨谱系匹配的区别

KLF6 在单核细胞中为正系数候选（+0.079），与其参与巨噬细胞炎症极化的实验研究具有方向上可讨论的联系。已有研究通过髓系细胞操作显示 KLF6 参与炎症相关基因调控 [5]。不过这些实验涉及巨噬细胞激活，不能直接外推为 HSC 向单核细胞分化的证据。血小板模型的 KLF6 系数更高（+0.081），其单扰动测试相关却仅为 0.177；现有文献 [5] 也不能证明它是巨核/血小板生成的驱动因子。该跨谱系匹配可能反映共有的应激或增殖相关表达成分，应进一步分解其贡献基因。

### 6.3 PAX8、OVOL2 与 RORC：需要优先排查背景效应

PAX8 在单核细胞中的标准化系数为 +0.100，但测试边际相关仅 0.110。PAX8 的经典表达与发育背景涉及甲状腺、肾脏等组织 [6]。这使其作为单核分化候选的解释具有不确定性：它可能在多变量模型中补偿相关扰动，而不是单核谱系特异调控。不能将同属 PAX 家族的 PAX8 与 B 细胞因子 PAX5 混淆；本文件没有 PAX5 扰动列。

OVOL2 在 CD4 T 与 NK 模型中均排名靠前，系数分别为 +0.116 与 +0.120，测试边际相关为 0.147 与 0.163。Ovol2 在乳腺上皮发育中限制上皮—间质转化的作用已有直接实验支持 [7]。这一背景与 MCF7 的乳腺来源密切相关，因此候选更可能首先提示源细胞背景的影响，而非已经发现新的 T/NK 分化开关。应检查匹配是否集中在上皮相关、细胞周期或广泛应激基因上。

RORC 在 B 细胞模型中作为正向代表候选，系数为 +0.042、测试边际相关为 0.135。RORC 编码的 RORγt 与 Th17 程序有关 [8]，但基因级扰动标签不能确定具体异构体，更不能据此推断它是 B 细胞命运决定因子。该结果属于有待排查的跨谱系统计关联。

### 6.4 从候选匹配到后续验证

可首先在独立的血液单细胞/供体数据上重复构建细胞平均谱，检查 MYB、KLF6 等候选是否仍入选；对 PAX8、OVOL2 等候选优先核实血液细胞中的表达和染色质可及性。随后按共表达模块划分基因进行验证，并扩大 alpha 搜索、增加重抽样次数、比较 Elastic Net，以评估相关扰动替代和候选排序的不确定性。当前 7 个目标的最小 CV MSE 位于搜索下界，这些扩展是后续工作，尚未完成。

实验上应在与目标相符的造血前体细胞中设置空载体/阴性对照、候选单扰动及必要的组合扰动，独立重复，检测表达程序、相应谱系表面标记以及存活/增殖，区分分化效应与一般细胞状态改变。红细胞与血小板的成熟阶段缺乏细胞核，其平均 RNA 特征不能被简单理解为可在成熟细胞中重新启动的转录程序。以上为验证设计建议，本次没有实施这些实验。

## 七、结论

在当前数据和固定验证设计下，L1000 扰动特征的稀疏线性组合能够重建部分血液细胞相对 HSC 的表达差异，但目标间差异明显。MYB 与红系相关程序的匹配具有较清楚的先验支持；KLF6 可与髓系激活文献联系，而 PAX8、OVOL2、RORC 等跨背景候选需要更谨慎验证。相近目标之间共享候选的现象支持表达程序的关联性，但不能证明共同的因果分化机制。最终证据应来自独立血液数据与相符细胞背景下的实验，而非仅靠系数排名或测试基因的 R²。

## 参考文献

[1] Subramanian A, et al. A Next Generation Connectivity Map: L1000 Platform and the First 1,000,000 Profiles. Cell. 2017;171:1437–1452.e17. DOI: 10.1016/j.cell.2017.10.049. https://pmc.ncbi.nlm.nih.gov/articles/PMC5990023/

[2] Lambert SA, et al. The Human Transcription Factors. Cell. 2018;172:650–665. DOI: 10.1016/j.cell.2018.01.029. https://pubmed.ncbi.nlm.nih.gov/29425488/

[3] Tibshirani R. Regression Shrinkage and Selection via the Lasso. Journal of the Royal Statistical Society, Series B. 1996;58:267–288. DOI: 10.1111/j.2517-6161.1996.tb02080.x.

[4] Mucenski ML, et al. A functional c-myb gene is required for normal murine fetal hepatic hematopoiesis. Cell. 1991;65:677–689. https://pubmed.ncbi.nlm.nih.gov/1709592/

[5] Date D, et al. Kruppel-like Transcription Factor 6 Regulates Inflammatory Macrophage Polarization. Journal of Biological Chemistry. 2014. https://pmc.ncbi.nlm.nih.gov/articles/PMC4036156/

[6] Poleev A, et al. PAX8, a human paired box gene: isolation and expression in developing thyroid, kidney and Wilms’ tumors. Development. 1992. https://pubmed.ncbi.nlm.nih.gov/1337742/

[7] Watanabe K, et al. Mammary Morphogenesis and Regeneration Require the Inhibition of EMT at Terminal End Buds by Ovol2 Transcriptional Repressor. Developmental Cell. 2014;29:59–74. DOI: 10.1016/j.devcel.2014.03.006. https://pmc.ncbi.nlm.nih.gov/articles/PMC4062651/

[8] Ivanov II, et al. The orphan nuclear receptor RORgammat directs the differentiation program of proinflammatory IL-17+ T helper cells. Cell. 2006;126:1121–1133. DOI: 10.1016/j.cell.2006.07.035. https://pubmed.ncbi.nlm.nih.gov/16990136/

## 复现与智能体使用说明

分析由用户提供目标并使用 Codex 智能体执行数据读取、程序编写、计算和文献检索。数值结果来自实际运行，未将 AI 生成的机制假说写成实验事实。关键提示词、运行与排错记录单独保存在随附的 Homework2_工作记录.md；PDF 已内嵌主要图表，无需另附图表才能阅读。可复现代码、依赖、完整系数和质控表保存在 https://github.com/chaigang713/CODEX/tree/main/analysis 。
'''
text=intro+base+'\n\n'+bio
captions={
'01_expression_distributions':'图1｜22种血液细胞的基因表达分布。箱线图隐藏离群点，使用文件提供的尺度。',
'02_zero_fraction':'图2｜各细胞类型零表达基因比例。髓系树突细胞约94%的零值提示信息量不足。',
'03_gene_heatmap':'图3｜60个高方差基因的聚类热图。颜色为逐基因跨细胞z分数，非原始表达量。',
'12_lineage_marker_heatmap':'图4｜预设血液谱系标记的表达热图。行不聚类，列采用Ward聚类，颜色为逐基因z分数。',
'05_cell_PCA':'图5｜2,000个高变基因的细胞PCA。红点为HSC；PC1与PC2分别解释27.4%和19.6%的变异。',
'06_test_R2':'图6｜21个目标的测试基因R²。红色标记高零比例的髓系树突细胞。',
'07_heldout_predictions':'图7｜六个代表目标的联合模型预测。横轴为观察差异，纵轴为预测差异；红虚线为一致线。',
'08_top_TFs':'图8｜六个目标的前10个TF。按系数绝对值排序；红色为正权重，蓝色为负权重。',
'10_TF_coefficients':'图9｜跨目标TF系数矩阵。展示在较多目标中入选的30个TF，颜色保留系数符号。',
'11_representative_TF_match':'图10｜单个代表TF特征与目标差异的测试基因匹配。六边形颜色表示基因密度，r为边际相关。',
'09_TF_overlap':'图11｜各目标前10个非零TF集合的Jaccard相似度。不包含系数方向，需与图9合看。'}
for name,cap in captions.items():text=re.sub(r'!\[[^\]]*\]\(figures/'+name+r'\.png\)',lambda m:m.group(0)+'\n\n'+cap,text)
(R/'SUBMISSION_REPORT.md').write_text(text)
pdfmetrics.registerFont(UnicodeCIDFont('STSong-Light'));styles=getSampleStyleSheet()
for name in ['Normal','Heading1','Heading2','Heading3']:
 styles[name].fontName='STSong-Light';styles[name].wordWrap='CJK'
styles['Normal'].fontSize=10.5;styles['Normal'].leading=17
styles['Heading1'].fontSize=19;styles['Heading1'].leading=27;styles['Heading1'].textColor=colors.HexColor('#153a50')
styles['Heading2'].fontSize=14;styles['Heading2'].leading=21;styles['Heading2'].spaceBefore=13;styles['Heading2'].textColor=colors.HexColor('#153a50')
styles['Heading3'].fontSize=12;styles['Heading3'].leading=19
small=ParagraphStyle('TableCell',parent=styles['Normal'],fontSize=8,leading=12)
caption=ParagraphStyle('Caption',parent=styles['Normal'],fontSize=9,leading=14,textColor=colors.HexColor('#395565'))
def safe(s):
 return html.escape(s).replace('−','-').replace('β','beta').replace('α','alpha').replace('₁','1').replace('²','^2').replace('ᵀ','T').replace('γ','gamma')
story=[]
for block in text.split('\n\n'):
 if not block.strip():continue
 if block.startswith('!['):
  name=re.search(r'\]\((.*?)\)',block).group(1);im=Image(str(R/name));scale=min(480/im.imageWidth,490/im.imageHeight);im.drawWidth=im.imageWidth*scale;im.drawHeight=im.imageHeight*scale;story.extend([im,Spacer(1,5)])
 elif block.startswith('|'):
  lines=[l for l in block.splitlines() if not re.match(r'^\|\s*---',l)]
  rows=[[Paragraph(safe(c.strip()),small) for c in l.strip('|').split('|')] for l in lines]
  n=len(rows[0]);widths=[480/n]*n
  if n==5:widths=[72,72,42,48,246]
  if n==6:widths=[100,76,76,76,76,76]
  tb=Table(rows,colWidths=widths,repeatRows=1,hAlign='LEFT');tb.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e8f0f4')),('GRID',(0,0),(-1,-1),.3,colors.HexColor('#cad5db')),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),4),('RIGHTPADDING',(0,0),(-1,-1),4)]));story.extend([tb,Spacer(1,12)])
 else:
  level=len(block)-len(block.lstrip('#'));style=styles['Heading1' if level==1 else 'Heading2' if level==2 else 'Heading3' if level==3 else 'Normal']
  if re.match(r'^图\d+｜',block):style=caption
  story.extend([Paragraph(safe(block.lstrip('# ')).replace('\n','<br/>'),style),Spacer(1,9)])
def footer(canvas,doc):
 canvas.setStrokeColor(colors.HexColor('#c5d3db'));canvas.line(55,37,540,37);canvas.setFont('STSong-Light',9);canvas.setFillColor(colors.HexColor('#4c6675'));canvas.drawString(55,24,'血液细胞与 L1000 扰动谱｜第二次作业');canvas.drawRightString(540,24,str(doc.page))
pdf=dest/'Homework2_分析报告.pdf'
SimpleDocTemplate(str(pdf),pagesize=(595,842),leftMargin=55,rightMargin=55,topMargin=46,bottomMargin=49,title='以HSC为参考的血液细胞与L1000扰动谱Lasso分析',author='').build(story,onFirstPage=footer,onLaterPages=footer)
log=(R/'WORK_LOG.md').read_text().replace('人工检查发现','智能体检查发现')
log=log.replace('## 用户关键指令','## 用户关键指令（依据对话整理）')
log+='''

## 代表性提示词与用户—智能体协作

下列为对话中的代表性指令摘录；上面的完整任务描述依据对话整理，未伪称单次逐字提示词。

1. “你先尝试能不能读取这些数据，若能，给我列出大致流程，我先审核一下。”智能体先检查文件结构和可读性，没有直接开始拟合；用户随后补充了分析要求。
2. “blood mean expr.csv这个是血液细胞表达数据的文件名，你再找一下。”首次及再次检查远端时未见文件；后续用户询问“现在呢”，重新fetch后发现并成功读取CSV，避免把本地旧快照当作远端最新内容。
3. “这是我的要求，你可以把这个要求拆分成几部分分别完成。”智能体据此执行质控、探索、模型及候选比较四部分，并报告中间发现与检查结果。
4. “我要交的有这两项，帮我完成。”用户提供课程提交要求截图后，智能体在已有实际结果上制作本次PDF和Markdown工作记录；补充研究问题、数据与方法、生物学解释、局限性、文献和内嵌主要图表。

## 文件取得与早期问题处理

微云访问先受到命令网络权限限制，申请命令网络权限后可访问GitHub，但微云仍返回403；未通过绕过代理下载。单个聊天附件传入执行环境有32MiB限制，因此用户上传六个25MiB左右分卷。智能体按aa到af合并，使用ZIP CRC检查和HDF5签名验证，解出134,243,972字节的L1000文件。因超过普通GitHub单文件限制，通过Git LFS上传。后续建模使用的文件SHA256为c8893967a5103a2abec9e0bb76602449bc65c75847de5f8064fa539d4293f826。

## 本次提交文件的补充整理与文献核对

本次没有重新拟合或更换主模型，也没有根据文献挑选更好看的模型结果。沿用已验证的21个模型、固定基因划分和全部指标。在原分析报告上补充摘要、研究问题、数据来源、明确的研究结论和图注，将主要图表直接嵌入PDF。

使用网页检索工具核对L1000、Lambert TF名单，以及MYB、KLF6、PAX8、OVOL2、RORC的论文摘要或可访问正文。文献用于解释已有结果，非额外计算结果。重点区分：MYB与造血/红系具有实验先验；KLF6的巨噬细胞激活证据不等于血小板分化证据；PAX8、OVOL2和RORC的跨背景入选需要谨慎解释。报告参考文献列出作者、标题和可追溯PMID/DOI链接；未宣称完成实验验证或穷尽文献。

核心文献入口：https://pubmed.ncbi.nlm.nih.gov/1709592/ 、https://pmc.ncbi.nlm.nih.gov/articles/PMC4036156/ 、https://pubmed.ncbi.nlm.nih.gov/1337742/ 、https://pmc.ncbi.nlm.nih.gov/articles/PMC4062651/ 、https://pubmed.ncbi.nlm.nih.gov/16990136/ 。

本次生成脚本为analysis/build_submission.py，负责读取已验证报告和图表、补充提交说明、排版中文PDF、生成此记录及只含两个必要文件的ZIP。图表编号和图注直接进入PDF，结果值与analysis/models/表格核对。

## 结果检查的归属与尚未完成的工作

本记录中的程序核对、图表查看和文献核对由智能体完成。用户已经提供文件、澄清数据名称并指定任务；本记录没有将用户尚未反馈的人工核查写成“本人已经检查”。建议用户提交前亲自阅读PDF，核对红细胞测试R²=0.621、MYB测试相关=0.350、单核细胞/经典单核细胞前10TF重叠等关键结果，再检查负系数和不符合预期的候选；这些属于提交前建议，不是已完成的用户操作。

尚未完成：独立供体验证、按基因模块分组验证、完整基因别名映射、扩大alpha搜索、增加重抽样次数、Elastic Net比较及血液细胞扰动实验。所有这些均在PDF中作为后续工作或局限说明，不伪装成已实施分析。

## 随附文件与复现位置

必要文件只有Homework2_分析报告.pdf与Homework2_工作记录.md。报告含主要图表，工作记录可独立阅读；没有强制依赖另附图表、原始大文件或代码。可选复现材料保存在https://github.com/chaigang713/CODEX/tree/main/analysis ，包括run_analysis.py、finalize_results.py、build_submission.py、requirements.txt、metadata.json、质控表和模型结果。此前的REPORT.pdf属于过程版本，正式提交以submission目录中的PDF为准。

学号、姓名和个人科研兴趣未由用户提供，因此未编造。可将ZIP按课程要求改名为“学号—姓名—Homework2.zip”后提交。本次未代用户发送邮件。
'''
(dest/'Homework2_工作记录.md').write_text(log)
zpath=R.parent/'Homework2_提交文件.zip'
with zipfile.ZipFile(zpath,'w',zipfile.ZIP_DEFLATED) as z:
 for p in [pdf,dest/'Homework2_工作记录.md']:z.write(p,p.name)
print(pdf);print(zpath)
for p in dest.iterdir():print(p.name,p.stat().st_size,hashlib.sha256(p.read_bytes()).hexdigest())
