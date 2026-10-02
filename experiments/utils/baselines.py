"""
Retrieval baselines compared with sheaf H^0: label-free (CCA, PLS, PCA,
whitened PCA) and label-requiring (LDA, NCA) subspaces of the same frozen
activations, plus external sentence encoders (SimCSE, SBERT) as reference
points.
"""

import numpy as np
import torch
from sklearn.cross_decomposition import CCA, PLSCanonical
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from transformers import AutoModel, AutoTokenizer

from sheafint.steering import joint_pca


def qr_basis(M: np.ndarray) -> np.ndarray:
    """Orthonormal basis of the column span of M."""
    Q, _ = np.linalg.qr(M)
    return Q


def lda_basis(H_train, labels, k, edge_dim):
    """LDA on fact labels inside the joint-PCA space."""
    P, _ = joint_pca(H_train, H_train, edge_dim)
    lda = LinearDiscriminantAnalysis(n_components=k, solver='eigen', shrinkage='auto')
    lda.fit(H_train @ P, labels)
    return qr_basis(P @ lda.scalings_[:, :k])


def cca_basis(A, B, k, edge_dim):
    """CCA between pair members inside the joint-PCA space."""
    P, _ = joint_pca(A, B, edge_dim)
    cca = CCA(n_components=k, max_iter=1000)
    cca.fit(A @ P, B @ P)
    return qr_basis(P @ cca.x_rotations_)


def pls_basis(A, B, k, edge_dim):
    """PLS-canonical between pair members inside the joint-PCA space."""
    P, _ = joint_pca(A, B, edge_dim)
    pls = PLSCanonical(n_components=k, max_iter=1000)
    pls.fit(A @ P, B @ P)
    return qr_basis(P @ pls.x_rotations_)


def pca_top_basis(H_train, k):
    """Top-k principal directions (orthonormal)."""
    return qr_basis(PCA(n_components=k).fit(H_train).components_.T)


def whitened_pca_map(H_train, k):
    """PCA whitening: top-k directions scaled by 1/sqrt(eigenvalue); apply to
    mean-centred activations. Returns (W, mu)."""
    pca = PCA(n_components=k).fit(H_train)
    W = pca.components_.T / np.sqrt(pca.explained_variance_)[None, :]
    return W, pca.mean_


def nca_basis(H_train, labels, k, edge_dim, seed):
    """Neighbourhood components analysis on fact labels inside the joint-PCA space."""
    P, _ = joint_pca(H_train, H_train, edge_dim)
    nca = NeighborhoodComponentsAnalysis(n_components=k, random_state=seed, max_iter=100)
    nca.fit(H_train @ P, labels)
    return qr_basis(P @ nca.components_.T)


@torch.no_grad()
def embed_simcse(texts, device, batch_size=32):
    """CLS-pooled embeddings from supervised SimCSE (Gao et al., 2021)."""
    name = 'princeton-nlp/sup-simcse-roberta-base'
    tok = AutoTokenizer.from_pretrained(name)
    enc_model = AutoModel.from_pretrained(name).to(device)
    enc_model.train(False)
    out = []
    for i in range(0, len(texts), batch_size):
        enc = tok(texts[i:i + batch_size], return_tensors='pt', padding=True,
                  truncation=True, max_length=64).to(device)
        out.append(enc_model(**enc).last_hidden_state[:, 0].float().cpu())
    del enc_model
    torch.cuda.empty_cache()
    return torch.cat(out, dim=0).numpy()


def embed_sentence_transformer(texts, device):
    """Embeddings from the all-mpnet-base-v2 sentence transformer."""
    from sentence_transformers import SentenceTransformer

    st = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device=device)
    emb = st.encode(texts, batch_size=64, show_progress_bar=False, convert_to_numpy=True)
    del st
    torch.cuda.empty_cache()
    return emb
