from scipy.stats import spearmanr, pearsonr

def get_spearman_corr(p, l):
    return spearmanr(p, l).correlation

def get_pearman_corr(p, l):
    return pearsonr(p, l)[0]