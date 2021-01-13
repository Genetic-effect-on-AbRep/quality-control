#!/bin/env python
import sys
import pandas

projects_matricesfn = sys.argv[1]

def run_manova(projects_matrices):
    print projects_matrices

def load_matrices(projects_matricesfn):
    projects_matrices = {}
    with open(projects_matricesfn,'r') as fh:
        for line in fh:
            line = line.rstrip().split('\t')
            project = line[0]
            matrixfn = line[1]
            projects_matrices[project] = pandas.read_csv(matrixfn,sep="\t",index_col=0)
    return projects_matrices

projects_matrices = load_matrices(projects_matricesfn)
manova = run_manova(projects_matrices)
