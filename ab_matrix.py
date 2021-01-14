#!/bin/env python
import sys
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import f_oneway

projects_matricesfn = sys.argv[1]
outdir = sys.argv[2]

def merge_dfs(projects_matrices):
    dfs = []
    for project in projects_matrices:
        df = projects_matrices[project]
        df['project'] = project
        dfs.append(df)
    df = pd.concat(dfs).dropna(axis='columns').reset_index(drop=True)
    return df
    
def run_pca(projects_matrices,outdir,statsfh,n_components=2):
    df = merge_dfs(projects_matrices)
    cols = list(df.columns)    
    cols.remove('project')
    cols.remove('Subject')
    x = df.loc[:, cols].values
    x = StandardScaler().fit_transform(x)
    y = df.loc[:,['project']].values
    pca = PCA(n_components=n_components)
    principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(data = principalComponents[:,:2],columns = ['principal component 1', 'principal component 2'])
    finalDf = pd.concat([principalDf, df[['project']]], axis = 1)
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_title('2 component PCA')
    targets = list(set(list(df['project'])))
    colors = ['r', 'g', 'b'][0:len(targets)]
    for target, color in zip(targets,colors):
        indicesToKeep = finalDf['project'] == target
        ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1'],
                   finalDf.loc[indicesToKeep, 'principal component 2'],
                   c = color,
                   s = 50)
        ax.legend(targets)
        ax.grid()
    fig.savefig("%s/pca.png" % outdir)
    plt.clf()
    for i,val in enumerate(pca.explained_variance_ratio_):
        statsfh.write("component-%s\t%s\tpca\n" % (i,round(val,3)))
    #print(pca.explained_variance_ratio_)

def plot_mean(projects_matrices,outdir):
    df = merge_dfs(projects_matrices)
    df1 = pd.melt(df,id_vars=['Subject','project'])
    g = sns.FacetGrid(df1,col='variable',col_wrap=10,sharey=False)
    g.map_dataframe(sns.boxplot,x='project',y='value')
    g.savefig("%s/genes.png" % outdir)
    plt.clf()

def run_anova_per_gene(projects_matrices,statsfh):
    df = merge_dfs(projects_matrices)
    projects = list(set(df['project']))
    out = []
    for col in df.columns:
        if col in ['Subject','project']:
            continue
        vals = []
        means = []
        for project in projects:
            col_project_vals = df.loc[df['project'] == project,col].values
            means.append(np.mean(col_project_vals))
            vals.append(col_project_vals)            
        f,p = f_oneway(*vals)
        out_line = [col,f,p] + means
        out.append(out_line)
    out.sort(key=lambda x: x[2])
    header = ["gene","f-stat","p-val"] + projects
    header.append("mean-anova")
    statsfh.write("%s\n" % "\t".join(header))
    for line in out:
        line.append("mean-anova")
        statsfh.write("%s\n" % "\t".join(map(str,line)))


def load_matrices(projects_matricesfn):
    projects_matrices = {}
    with open(projects_matricesfn,'r') as fh:
        for line in fh:
            line = line.rstrip().split('\t')
            project = line[0]
            matrixfn = line[1]
            df = pd.read_csv(matrixfn,sep="\t")#,index_col=0)
            subjects = df['Subject']
            ighv_cols = df[df.filter(like='IGHV').columns]
            projects_matrices[project] = pd.concat([subjects,ighv_cols],axis=1)            
    return projects_matrices

def plot_cols(projects_matrices,outdir):
    for project in projects_matrices:
        df = projects_matrices[project]
        df1 = pd.melt(df,id_vars=['Subject'])
        g = sns.FacetGrid(df1,col='Subject',col_wrap=10,sharey=False)
        g.map_dataframe(sns.barplot,x='variable',y='value')
        g.savefig("%s/%s.png" % (outdir,project))
        plt.clf()

def plot_no_val_counts(projects_matrices,outdir,statsfh):
    for project in projects_matrices:
        df = projects_matrices[project]
        df_vals = df.iloc[:,1:]
        df_counts = pd.concat([df['Subject'],df_vals[df_vals == 0].count(axis=1)],axis=1).sort_values(by=[0]).reset_index(drop=True)
        plot_row_stats(df_counts,"Number of IGHV genes with 0 value","%s/%s_zero_count.png" % (outdir,project))
        df_counts['project'] = project
        df_counts['stat'] = 'zero_count'
        df_counts.to_csv(statsfh,mode='a',sep='\t',index=False)

def plot_max_val(projects_matrices,outdir,statsfh):
    for project in projects_matrices:
        df = projects_matrices[project]
        df_vals = df.iloc[:,1:]
        df_max = pd.concat([df['Subject'],df_vals.max(axis=1)],axis=1).sort_values(by=[0]).reset_index(drop=True)
        plot_row_stats(df_max,"Max value","%s/%s_max_val.png" % (outdir,project))
        df_max['project'] = project
        df_max['stat'] = 'max_value'
        df_max.round(3).to_csv(statsfh,mode='a',sep='\t',index=False)
        

def plot_row_stats(df_stat,ylabel,fn):
    width = df_stat.shape[0]/3.1
    plt.figure(figsize=(width,4))
    plot = sns.barplot(x = 'Subject',
                       y = 0,
                       data=df_stat,
                       color="salmon",
                       order=df_stat['Subject'])
    plot.set(ylabel=ylabel,xlabel="Subject")
    plot.get_figure().savefig(fn)
    plt.clf()
    
statsfh = open("%s/stats.txt" % outdir,'w')
projects_matrices = load_matrices(projects_matricesfn)

plot_cols(projects_matrices,outdir)
plot_no_val_counts(projects_matrices,outdir,statsfh)
plot_max_val(projects_matrices,outdir,statsfh)
run_pca(projects_matrices,outdir,statsfh)
run_anova_per_gene(projects_matrices,statsfh)
plot_mean(projects_matrices,outdir)

statsfh.close()
