#!/usr/bin/python
import pandas as pd
import numpy as np
import subprocess
import sys
import os
from glob import iglob
from textwrap import wrap
from copy import deepcopy
from datetime import datetime
import matplotlib
import platform
if platform.system() == 'Darwin':
    matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from evcouplings import align
from scipy.stats import gaussian_kde
from evcouplings.couplings import CouplingsModel

'''
Contents:
    - scripts to load biological data types into dataframes
    - scripts to check for conserved regions of alignment
    - and to plot the predicted fitness across natural / optimized sequences


'''

HMMSEARCH = '/n/groups/marks/software/jackhmmer/hmmer-3.1b1-linux-intel-x86_64/binaries/hmmsearch'
ESL_REFORMAT = '/n/groups/marks/software/jackhmmer/hmmer-3.1b1-linux-intel-x86_64/binaries/esl-reformat'
UNIREF = '/n/groups/marks/databases/jackhmmer/uniref100/uniref100_current.o2.fasta'

# file parsing #

def read_fa(fa_file):
    '''reads fasta file into header/sequence pairs'''
    header = ''
    seq = ''
    seqs = []
    with open(fa_file, 'r') as f:
        for line in f:
            line = line.strip()
            if len(line) == 0 or line[0] == '#':
                continue
            if line[0] == '>':
                seqs.append((header, seq))
                header = line[1:]
                seq = ''
            else:
                seq += line
    seqs.append((header, seq))
    seqs = pd.DataFrame(seqs[1:], columns=['header', 'seq'])
    seqs.loc[:, 'seq_ID'] = seqs['header'].apply(lambda x: x.split(' ')[0])
    seqs.loc[:, 'species'] = seqs['header'].apply(
        lambda x: x.split('Tax=')[1].split('TaxID=')[0]
                  if 'Tax=' in x else None)
    return(seqs)

def write_fa(fa_table, outfile):
    '''write seq table to fasta'''
    with open(outfile, 'w') as f:
        for i, row in fa_table.iterrows():
            f.write('>' + row['header'] + '\n')
            f.write(row['seq'] + '\n')

def get_domain(seq, domain=(1, -1), start=1, incl=1):
    '''retrieve sequence domain, within provided coords
    if provided domain is inclusive, specify incl=1'''
    a, b = (domain[0] - start, domain[1] - start + incl)
    return(seq[a:b])

def trim_fa(fa_file, domain, start=1, outfile=None):
    '''trim alignment to specified sub-region'''
    if isinstance(fa_file, str):
        ali = read_fa(fa_file)
    else:
        ali = deepcopy(fa_file)
        
    ali.loc[:, 'seq'] = ali['seq'].apply(
        lambda x: get_domain(x, domain, start))

    if isinstance(outfile, str):
        write_fa(ali, outfile)
    return(ali)

def aligned_region(seq, aas='.-ACDEFGHIKLMNPQRSTVWY'):
    '''removes 'unaligned' lowercase regions from sequences '''
    aligned = seq.rstrip(aas.lower())
    return(aligned)

def hmmscan_parse(hmmscanfile):
    '''parses hmmscan file into a dataframe'''
    filelen = sum(1 for line in open(hmmscanfile, 'r'))
    skiplines = set(filelen - np.arange(0, 11))
    skiplines.update([0, 1, 2])
    
    header = [
        'target name','pfam','tlen','query name',
        'accession','qlen','E-value','score','bias','#','of',
        'c-Evalue','i-Evalue','score_2','bias','from-hmm','to-hmm','from',
        'to','from-env','to-env','acc','description']
    numerics = [header[2]]+header[5:-2]

    rows = []
    with open(hmmscanfile, 'r') as file:
        for i, line in enumerate(file):
            if i == 1:
                split_at = len(line.split('accession')[0])
            if i in skiplines:
                continue

            row = [line[:split_at].rstrip()]  + line[split_at:].rstrip().split(' ')
            row = [r for r in row if (r != '')]
            row = row[:22] + [' '.join(row[22:])]
            rows.append(row)

    table = pd.DataFrame(rows, columns=header)
    table.loc[:, numerics] = table[numerics].apply(pd.to_numeric)
    table.loc[:, 'seq_ID'] =  table.apply(
        lambda x: x['target name']+'/'+str(int(x['from']))+'-'+str(int(x['to'])), axis=1)
    return(table)

# sequence feature computation #

def seq_features(seq, aa_feature, aa_s='ACDEFGHIKLMNPQRSTVWY'):
    '''converts amino acid sequence into corresponding properties'''
    if aa_s is not None:
        seq_f = [s for s in seq if s in set(aa_s)]
    return(np.array([aa_feature.get(s, np.nan) for s in seq]))

def helix_moment(seq_feats, normed=False, t_step=100):
    '''computes magnitude of the moment around a helix for given values'''
    theta = np.arange(len(seq_feats))*(t_step*np.pi/180)
    x = np.sum(np.sin(theta)*seq_feats)
    y = np.sum(np.cos(theta)*seq_feats)
    u = np.sqrt(x**2 + y**2)
    if normed:
        u = u / np.sum(np.abs(seq_feats))
    return(np.abs(u))


# compile information #

def score_seqs(hmm_dmtbl, hmm_aln, evh_mdl, region=None, model_aln=None, seqcol='seq'):
    '''scores sequences and attributes species'''
    # map fasta info to domtbl
    hmm_dmtbl = pd.merge(hmm_dmtbl, hmm_aln, on='seq_ID')
    
    # optional - cut out region and label seqs from model
    if region is not None:
        hmm_dmtbl.loc[:, seqcol+'_domain'] = hmm_dmtbl[seqcol].apply(
            lambda x: x[region[0]:region[1]] if isinstance(x, str) else '')
        seqcol = seqcol+'_domain'
    if model_aln is not None:
        model_seqs = set(model_aln['seq_ID'])
        hmm_dmtbl.loc[:, 'in_model'] = hmm_dmtbl['seq_ID'].isin(model_seqs)
    
    # score sequences by EVH model
    seqs = hmm_dmtbl[seqcol]
    scorable = seqs.apply(lambda x: len(x)==len(evh_mdl.target_seq) and
        set(x).issubset(set('ACDEFGHIKLMNPQRSTVWY')))
    print('scorable seqs: ', sum(scorable), '%', 100*np.mean(scorable))
    hmm_dmtbl.loc[scorable, 'EVH'] = seqs[scorable].apply(
        lambda x: evh_mdl.hamiltonians([x])[0][0])
    hmm_dmtbl.loc[~scorable, 'EVH'] = np.nan

    WT_score = evh_mdl.hamiltonians([''.join(evh_mdl.target_seq)])[0][0]
    hmm_dmtbl.loc[:, 'delta_EVH'] = hmm_dmtbl['EVH'] - WT_score

    return(hmm_dmtbl)

def filt_seq(seq, L=None):
    '''toss out inserts relative to model (lowercase in a2m format from stockholm)'''
    seq_f = [l for l in seq if l not in set('acdefghiklmnpqrstuvwxyz.')]
    if L is not None and len(seq_f)!=L:
        return(None)
    return(''.join(seq_f))

def skinny_a2m(a2m_file, outfile=None):
    a2m = read_fa(a2m_file)
    a2m.loc[:, 'seq'] = a2m['seq'].apply(filt_seq)
    if outfile is None:
        outfile = a2m_file.replace('.a2m', '-skinny.a2m')
    write_fa(a2m, outfile)

# comparison plots #
colorkit=['pink','violet','green','yellow','blue','red','salmon','tan','hotpink', 'black', 'purple']
def energy_histograms(hmm_dmtbl, extra_seqs=None, xt='EVH',
                      sele=[], sele_col='species', bins=150,
                      cs=colorkit, outfile='temp-hist.png'):
    '''plot histogram of sequence results, inside/outside model alignment'''
    fig, ax = plt.subplots(1)
    fig.set_size_inches(10, 5)
    ax.set_ylabel('# of seqs')
    ax.set_xlabel(xt)
    
    bins = ax.hist(hmm_dmtbl[xt].dropna(), bins=bins,
                   color='lightgray', lw=0.5, edgecolor='k')
    fig.savefig(outfile.replace('.png','-basic_dist.png'), dpi=200)

    if isinstance(extra_seqs, str):
        ax.hist(hmm_dmtbl[hmm_dmtbl[extra_seqs]][xt].dropna(), bins=bins[1],
                color='c', lw=0.5, edgecolor='k')
    elif isinstance(extra_seqs, pd.DataFrame):
        ax.hist(extra_seqs[xt].dropna(), bins=bins[1],
                color='c', lw=0.5, edgecolor='k')        
    elif isinstance(extra_seqs, dict):
        for c, vals in extra_seqs.items():
            ax.hist(vals.dropna(), bins=bins[1],
                    color=c, lw=0.5, edgecolor='k')
    
    spec_report = []
    Ls = []
    for s, c in zip(sele, cs):
        spec = hmm_dmtbl[hmm_dmtbl[sele_col].str.contains(s, na=False)]
        spec_report.append(spec)
        Ls.append(len(spec))
        ax.hist(spec[xt].dropna(), bins=bins[1],
               color=c, lw=1, edgecolor=c, alpha=0.8)
    ax.set_title('\n'.join(wrap(str(list(zip(sele, cs, Ls))), 150)), fontsize=6)
    fig.tight_layout()
    if outfile is not None:
        fig.savefig(outfile, dpi=200)
    
    return(pd.concat(spec_report))


def cut_inserts(alignment_file):
    '''takes alignment file and ruthlessly strips out inserts (non-caps)'''
    trimmed_file = '.'.join(alignment_file.split('.')[:-1])+'_insert-trimmed.a2m'
    
    newf = []
    with open(alignment_file,'r') as f:
        for line in f:
            if line[0] == '>':
                newf.append(line)
            else:
                newf.append(''.join([a for a in line
                     if not a in set('abcdefghijklmnopqrstuvwxyz')]))

    with open(trimmed_file, 'w') as f:
        for line in newf:
            f.write(line)

    return(trimmed_file)


def smooth_window(data, n=5):
    '''rolling average'''
    z = [np.mean(data[i:i+n]) for i in range(len(data)-n)]
    return(z)


def conservation_map(myaln, n=3, weights=True, gap_renorm=False, outfile=None):
    with open(myaln, 'r') as f:
        aln = align.Alignment.from_file(f, format='fasta', a3m_inserts='first')
    
    if weights:
        aln.set_weights
    
    remap = {i: s for i, s in enumerate(aln.alphabet)}
    frqs = deepcopy(aln.frequencies)
    
    if gap_renorm:
        aln.frequencies[:, 0] = 10**-10
        aln.frequencies[:, :] = aln.frequencies / aln.frequencies.sum(axis=1)[:, np.newaxis]
    
    cons = np.argmax(aln.frequencies, axis=1)
    cons = ''.join([remap[s] for s in cons])
        
    fig, axs = plt.subplots(2, 1)
    fig.set_size_inches(20, 10)
    
    # conservation per site
    axs[0].scatter(np.arange(aln.L), aln.conservation(), s=4, c='k')
    axs[0].plot(np.arange(aln.L-n)+n/2, smooth_window(aln.conservation(), n))
    axs[0].set_xticks(np.arange(aln.L)[::25])
    
    # aa freq heatmap
    cm = axs[1].imshow(frqs.T, aspect='auto')
    fig.colorbar(cm, pad=0.02, fraction=0.005)
    axs[1].set_yticks(np.arange(21))
    axs[1].set_yticklabels(aln.alphabet)
    axs[1].set_xticks(np.arange(aln.L)[::25])
    fig.tight_layout()
    if isinstance(outfile, str):
        fig.savefig(outfile, dpi=200)
    
    return(cons, aln)

def plot_surface(seqs, aa_prop, coords=None, cm='Spectral', vm=None, vM=None,
                aa_s=None, figax=None, size=0.33, outfile=None,
                t_step=100, c=11):
    '''plot surface of helices, spread out left-to-right'''
    if isinstance(seqs, str):
        seqs = wrap(seqs, c)
        seqs[-1] += '-'*(c-len(seqs[-1]))
        
    c, n = len(seqs[0]), len(seqs)
    if coords is None:
        coords = helix_pattern(c, t_step=t_step)
    elif coords is 'normal':
        coords = np.arange(c)
    if figax is None:
        fig, ax = plt.subplots(1)
        fig.set_size_inches(c*size, n*size)
    else:
        fig, ax = figax
    
    v = np.array([seq_features(seq, aa_prop, aa_s=aa_s) for seq in seqs])
    ax.imshow(v[:, coords], cmap=cm, vmin=vm, vmax=vM)
    ax.set_xticks(range(c))
    ax.set_xticklabels(coords)
    ax.set_yticks(range(n))
    ax.set_yticklabels(np.arange(n)*c+1)
    if outfile is not None:
        fig.savefig(outfile, dpi=200)
    return(v)

def helix_pattern(n, t_step=100):
    '''constructs helical indices'''
    return(np.argsort((np.arange(n)*t_step)%360))

def stats_from_sto(sto_file):
    ''''''
    stats = {'len': [], 'insert': [], 'unaligned_insert': []}
    with open(sto_file, 'r') as f:
        for line in f:
            if line[0] == '#' and '/' in line:
                line = line.strip()
                ali = line.split()[-1]
                stats['len'].append(len(ali))
                stats['insert'].append(ali.count('.') + ali.count('-'))
                stats['unaligned_insert'].append(ali.count('~'))
                
    return()
 
def cmd(command, timeout=30*60):
    '''run terminal commands'''
    print(' '.join(command))
    proc = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        o, e = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        o, e = proc.communicate()

    if e:
        print('Output: ' + o.decode('ascii'))
        print('Error: '  + e.decode('utf-8'))
        print('code: ' + str(proc.returncode))

    return(o.decode('ascii'))
        
def validate_ali(target_fa, hmm):
    '''find # of aligned residues, and bitscore of match-
    compare to included and excluded model seqs'''
    ali = cmd([HMMSEARCH, hmm, target_fa])
    ali = hmmsearch_parse(ali)
    top_ali = ali.iloc[0]
    top_ali = {  
        'hmm_consensus': top_ali['consensus'],
        'hmm_matches': top_ali['match'],
        'hmm_probabilities':  top_ali['probability'],
        'indices_aligned': top_ali['i_hmm'] == top_ali['i_target'],
        'covered_50p': sum([(a in set('56789*')) for a in top_ali['probability']]),
        'covered_90p': sum([(a in set('9*')) for a in top_ali['probability']]),
        'identity': sum([(a not in set('+ ')) for a in top_ali['match']]),
        'identity_blo': sum([(a not in set(' ')) for a in top_ali['match']]),
        'E-value': top_ali['E']
    }
    return(top_ali)


def summarize_ali(evc_run, outdir=None):
    ''''''
    evc_run = evc_run.strip('/')
    evc_id = evc_run.split('/')[-1]
    
    if (outdir is None) or outdir == 'assessment':
        outdir = evc_run + '/assessment'
        mkdir(outdir)

    # inputs
    target_fa = '/'.join([evc_run, 'align', evc_id+'.fa'])
    model_params = '/'.join([evc_run, 'couplings', evc_id+'.model'])
    checkpoints = '/'.join([evc_run, 'align', evc_id+'-*'])
    domtbl = '/'.join([evc_run, 'align', evc_id+'.domtblout'])
    model_a2m= '/'.join([evc_run, 'align', evc_id+'.a2m'])
    
    ali_stats = pd.DataFrame()
    for checkpoint in iglob(checkpoints):
        ext = checkpoint.split('.')[-1]
        i = checkpoint.split('-')[-1].split('.')[0]
        if ext == 'sto':
            a2m = outdir+'/'+checkpoint.split('/')[-1].replace('.sto', '.a2m')
            s2m = a2m.replace('.a2m', '-skinny.a2m')
            cmd([ESL_REFORMAT, '-o', a2m, 'a2m', checkpoint])
            skinny_a2m(a2m)
            for t, g in zip(['incl_gaps', 'excl_gaps'], [False, True]):
                cons_plot = outdir+'/'+evc_id+'-a2m-'+t+'-conservation-'+i+'.png'
                cons, ali = conservation_map(
                    s2m, gap_renorm=g, outfile=cons_plot)
                ali_stats.loc[i, 'couplings_model_consensus-'+t] = cons
            ali_stats.loc[i, 'N_seqs'] = ali.N
        elif ext == 'hmm':
            target_comp = validate_ali(target_fa, checkpoint)
            for k, v in target_comp.items():
                ali_stats.loc[i, k] = v
    
    ali_stats.sort_index().to_csv(outdir+'/'+evc_id+'-jackhmmer_iteration_stats.csv')
    

ASSESS_SCRIPT='''
python3.6 SCRIPT 'summarize' EVC OUT
HMMSEARCH
python3.6 SCRIPT 'selfassess' EVC DEEP_ALI
'''

def full_assessment(evc_run, myscript=None, outdir=None, t='2:00:00'):
    ''''''
    # we'll submit everything as one sbatch bash script
    if myscript is None:
        myscript = sys.argv[0]

    # by default we want to create an assessment directory
    evc_run = evc_run.strip('/')
    evc_id = evc_run.split('/')[-1]
    if (outdir is None) or (outdir == 'assessment'):
        outdir = evc_run + '/assessment'
        mkdir(outdir)

    # find the hmm in the last iteration of the align stage
    final_hmm = sorted(iglob(evc_run+'/align/'+evc_id+'*.hmm'))[-1]
    deep_ali = outdir+'/'+evc_id+'_E1000_hmm'
    hmm_job = HMMsearch(final_hmm, deep_ali, submit=False)
    # if the search is already performed, skip
    if not os.path.exists(hmm_job['aln']):
        hmm_str = ' '.join([str(a) for a in hmm_job['command']])
    else:
        hmm_str = '\n'
        t = '30:00'
    # prepare the bash script
    cmd = ASSESS_SCRIPT.replace('SCRIPT', myscript)
    cmd = cmd.replace('EVC', evc_run)
    cmd = cmd.replace('OUT', outdir)
    cmd = cmd.replace('HMMSEARCH', hmm_str)
    cmd = cmd.replace('DEEP_ALI', deep_ali)
    sbatch(cmd, mem='20G', t=t, p='short', jobname=outdir+'/')


def selfassess_evh(evc_run, deep_ali=None, outdir=None):
    '''find potential homologs by the final hmm produced in align stage,
    score these and those deemed 'functional homologs' and used to build the model,
    compare these and to the target sequence'''
    evc_run = evc_run.strip('/')
    evc_id = evc_run.split('/')[-1]
    
    if (outdir is None) or (outdir == 'assessment'):
        outdir = evc_run + '/assessment'
        mkdir(outdir)
    if (deep_ali is None) or (deep_ali == 'find'):
        deep_ali = sorted(iglob(evc_run+'/align/'+evc_id+'*.sto'))[-1]
        print(deep_ali)
        deep_ali = deep_ali.replace('.sto','')

    #inputs
    deep_domtbl = deep_ali+'.domtblout'
    deep_sto = deep_ali+'.sto'
    model_params = '/'.join([evc_run, 'couplings', evc_id+'.model'])
    model_a2m= '/'.join([evc_run, 'align', evc_id+'.a2m'])
    #outputs
    deep_a2m = outdir + '/' + deep_ali.split('/')[-1]+'.a2m'
    deep_s2m = deep_a2m.replace('.a2m','-skinny.a2m')

    # prepare job for sel
    deep_domtbl = hmmscan_parse(deep_domtbl)
    model_ali = read_fa(model_a2m)
    model = CouplingsModel(model_params)
    print('model target sequence:',''.join(model.target_seq))

    # need to keep just the sequence within EVH-modeled range
    target_seq = model_ali['seq'].iloc[0]
    model_region = modelled_region(target_seq)
    print('extracted target sequence:',target_seq)
    
    # convert and trim the target alignment
    if not os.path.exists(deep_a2m):
        cmd([ESL_REFORMAT, '-o', deep_a2m, 'a2m', deep_sto])
    if not os.path.exists(deep_s2m):
        skinny_a2m(deep_a2m)

    deep_ali = read_fa(deep_s2m)
    deep_ali.loc[:, 'seq'] = deep_ali['seq'].apply(
        lambda x: get_domain(x, model_region))
    print('extracted target sequence, trimmed:',deep_ali['seq'].iloc[0])

    deep_domtbl = score_seqs(deep_domtbl, deep_ali, model, model_aln=model_ali)

    for xt in ['EVH', 'delta_EVH']:
        sp_hits = energy_histograms(
            deep_domtbl, extra_seqs='in_model', xt=xt,
            sele=SELE_SPECIES, sele_col='species', bins=150,
            outfile=outdir+'/'+xt+'-model_seqs_vs_distant-hist.png')
    deep_domtbl.loc[:, 'target_identity'] = deep_domtbl['seq'].apply(
        lambda x: np.sum(np.array(
            [(a == b) for a, b in zip(x, target_seq)], dtype=bool)))
    
    sp_hits.to_csv(outdir+'/species_hits_w_scores.csv')
    deep_domtbl.to_csv(outdir+'/deepsearch_w_scores.csv')
    evh_vs_ali(deep_domtbl, outdir+'/evh-vs-ali.png')

SELE_SPECIES = [
'Leishmania', 'elegans', 'radiodurans', 'Adineta', 'Hypsibius',
'Ramazzottius', 'vanderplanki', 'superbus', 'lepidophylla', 'Plasmod', 'sapiens']

def evh_vs_ali(scored_domtbl, outfile='temp.evh-vs-ali.png'):
    '''takes domtbl with pre-computed EVH scores and plots
    various comparisons of the 'fitness' and the alignment confidence'''
    fig, axs = plt.subplots(1, 3)
    fig.set_size_inches(10, 3)
    scored_domtbl.loc[:, '-logE'] = scored_domtbl['E-value'].apply(
        lambda x: -np.log(x))
    for ax, y in zip(axs, ['-logE', 'score', 'target_identity']):
        with pd.option_context('mode.use_inf_as_null', True):
            to_plot = ~scored_domtbl['EVH'].isna() & ~scored_domtbl[y].isna()
        z = scored_domtbl[to_plot]
        scatter_density(
            z['EVH'].values, z[y].values, ax)
        ax.set_xlabel('EVH')
        ax.set_ylabel(y)

        if sum(z['in_model']) > 0:
            z_m = z[z['in_model']]
            ax.scatter(
                z_m['EVH'].values, z_m[y].values, c='c', alpha=0.25)

    fig.tight_layout()
    fig.savefig(outfile, dpi=250)




def modelled_region(seq):
    '''given seq, retrieve region of alignment included in couplings model,
    this is from the first to last uppercase'''
    region = []
    for i, aa in enumerate(seq):
        if aa in set('ABCDEFGHIJKLMNOPQRSTUVWXYZ'):
            region.append(i)
    region = (region[0]+1, region[-1]+1)
    return(region)


def mkdir(dirname):
    if not os.path.isdir(dirname):
        cmd(['mkdir',dirname])


def hmmsearch_parse(output):
    '''parse hmmsearch output to dataframe'''
    cols = ['struct', 'consensus', 'match', 'target', 'probability', 'i_hmm', 'i_target', 'E']
    data = []
    out = output.split('\n\n\n\n')[0]
    for line in out.split('==')[1:]:
        domain = ['', '', '', '', '', [], []]
        rows = line.split('\n')
        header = rows[0]
        body = '\n'.join(rows[1:])
        ev = float(header.split(': ')[-1])
        for j, l in enumerate(body.split('\n\n')[:-1]):
            x = [a.strip() for a in l.split('\n')[:5]]
            if len(x) == 4: 
                c, m, t, p = [a.strip() for a in l.split('\n')]
                s = ''
            else:
                s, c, m, t, p = [a.strip() for a in l.split('\n')[:5]]
                s = s.split(' ')[0]
            i_h = [c.split(' ')[-1]]
            i_t = [t.split(' ')[-1]]
            p = p.split(' ')[0]
            c = c.split(' ')[-2]
            t = t.split(' ')[-2]
            
            for i, v in enumerate([s, c, m, t, p, i_h, i_t]):
                domain[i] += v
        domain.append(ev)
        data.append(domain)
    data = pd.DataFrame(data, columns=cols).sort_values('E')
    return(data)

JOBFMT = '''
#SBATCH -J 
#SBATCH -o 
#SBATCH -e 
#SBATCH -t 
#SBATCH -p 
#SBATCH --mem=
'''.strip().split('\n')

def sbatch(cmd_str, p='short', t='4:00:00', mem='20G', jobname='', o=None, e=None):
    if isinstance(cmd_str, list):
        cmd_str = ' '.join([str(a) for a in cmd_str])
    current_time = datetime.today().isoformat()
    jobname = jobname + current_time.replace(':', '-')+'.job.txt'
    if o is None:
        o = jobname.replace('.job', '.output')
    if e is None:
        e = jobname.replace('.job', '.errors')

    with open(jobname, 'w') as f:
        f.write('#!/bin/bash -l\n')
        for s, v in zip(JOBFMT, [jobname, o, e, t, p, mem]):
            f.write(s+str(v)+'\n')
        f.write(cmd_str)

    print('submitting job ('+jobname+'): ', cmd_str)
    out = cmd(['sbatch', jobname])
    return(out)


def HMMsearch(hmm, out, seqdb=UNIREF, E=1000, submit=True, **kwargs):
    txt = out+'.out.txt'
    aln = out+'.sto'
    tbl = out+'.tblout'
    dmtbl = out+'.domtblout'

    cmd = [HMMSEARCH, '-o', txt, '-A', aln,
           '--tblout', tbl, '--domtblout', dmtbl,
           '-E', E, '--incE', E, hmm, seqdb]
    if submit:
        o = sbatch(cmd, **kwargs)
        print(o)

    return({'command': cmd, 'out': txt, 'aln': aln, 'tbl': tbl, 'domtbl': dmtbl})

def msa_weights(oh, theta=0.8, pseudocount=0):
    #i = pairwise_identity(oh)
    #neighbors = msa_neighbors(oh, theta, pseudocount)
    neighbors = align.alignment.num_cluster_members(oh, theta)
    #(i > theta).sum(0)/2
    w = 1 / (neighbors + pseudocount)
    return(w, neighbors)


def scatter_density(x, y, ax):
    '''creates scatter plot, coloring dots by density in area'''
    # Calculate the point density
    x, y = np.array(x), np.array(y)
    xy = np.vstack([x,y]).astype(float)
    z = gaussian_kde(xy)(xy)

    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]
    
    ax.scatter(x, y, c=z, s=50, edgecolor='', cmap='viridis')

a2n = dict([(a, n) for n, a in enumerate('ACDEFGHIKLMNPQRSTVWY-')])
    
def encode_aa(seq, a2n):
    '''numerically encode string'''
    aas = set(a2n.keys())
    return([a2n[a] if (a in aas) else a2n['-'] for a in seq])

def onehot_fa(fa):
    '''convert sequences to fasta'''
    if isinstance(fa, pd.DataFrame):
        seqs = fa['seq']
    else:
        seqs = fa
        
    msa = []
    for seq in seqs:
        msa.append(encode_aa(seq, a2n))
    
    oh = onehot(np.array(msa))
    return(oh)
    
def onehot(labels, N):
    '''one-hot encode a numpy array'''
    O = np.reshape(np.eye(N)[labels], (*labels.shape, N))
    return(O)
    

if __name__ == "__main__":
    if sys.argv[1] == 'summarize':
        summarize_ali(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == 'hmmsearch':
        HMMsearch(sys.argv[2], sys.argv[3], mem='20G', t='4:00:00')
    elif sys.argv[1] == 'selfassess':
        if len(sys.argv) > 4:
            OUTDIR = sys.argv[4]
        else:
            OUTDIR = None
        selfassess_evh(sys.argv[2], sys.argv[3], OUTDIR)
    elif sys.argv[1] == 'full':
        full_assessment(sys.argv[2], sys.argv[0])