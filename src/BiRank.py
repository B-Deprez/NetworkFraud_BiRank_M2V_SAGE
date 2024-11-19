import pandas as pd
from scipy import sparse
import numpy as np

def SNMM(adjMat):
    Dp = sparse.diags(np.asarray(1 / np.sqrt(adjMat.sum(axis=1).flatten())), [0])
    Dc = sparse.diags(np.asarray(1 / np.sqrt(adjMat.sum(axis=0))), [0])
    S_t = Dp @ adjMat @ Dc
    S = S_t.transpose()
    return(S)

def BiRank(Network, claim_nodes, party_nodes, fraudMat, alpha = 0.85, maxiter = 1000, eps = 1e-14):
    claim_nodes = claim_nodes.index
    party_nodes = party_nodes.index
    c0 = fraudMat/sum(fraudMat['FraudInd'])
    c = []
    p = []

    HG = list(party_nodes)+list(claim_nodes)

    if not isinstance(Network, sparse.csr.csr_matrix):
        adjMat = Network.to_adjacency_matrix(HG)[:len(party_nodes), len(party_nodes):]
    else:
        adjMat = Network

    S = SNMM(adjMat)

    pOld = np.random.uniform(size = adjMat.shape[0])
    cOld = np.random.uniform(size = adjMat.shape[1])

    iter = 0
    cont_eps = True
    while (iter < maxiter) & (cont_eps):
        A = (S @ pOld)
        A.shape = (len(claim_nodes), 1)
        c_t = alpha * A + (1 - alpha) * c0.values
        c = c_t.transpose()
        p = (c @ S)

        eps_c = np.sqrt((c - cOld) @ (c - cOld).transpose()) / np.sqrt(cOld @ cOld.transpose())
        eps_c = eps_c[0][0] > eps

        eps_p = np.sqrt((p - pOld) @ (p - pOld).transpose()) / np.sqrt(pOld @ pOld.transpose())
        eps_p = eps_p[0][0] > eps
        cont_eps = eps_c & eps_p

        cOld = c[0]
        pOld = p[0]
        iter += 1

        if not cont_eps:
            print("Convergence reached.")

        if iter == maxiter:
            print("Maximal iteration reached.")

    c=c[0]
    p=p[0]

    ResultsClaims = {"ID" : np.arange(adjMat.shape[1]),
                     "Score" : c,
                     "StdScore": (c-np.mean(c))/np.std(c, ddof=1),
                     "ScaledScore" : (c-np.min(c))/(np.max(c)-np.min(c))}
    ResultsClaims = pd.DataFrame(data = ResultsClaims)
    ResultsClaims = ResultsClaims.sort_values(by = "Score", ascending= False)

    ResultsParties = {"ID" : np.arange(adjMat.shape[0]),
                     "Score" : p,
                     "StdScore": (p-np.mean(p))/np.std(p, ddof=1),
                     "ScaledScore" : (p-np.min(p))/(np.max(p)-np.min(p))}
    ResultsParties = pd.DataFrame(data = ResultsParties)
    ResultsParties = ResultsParties.sort_values(by="Score", ascending= False)

    return(ResultsClaims,
           ResultsParties,
           adjMat,
           iter,
           iter < maxiter)

