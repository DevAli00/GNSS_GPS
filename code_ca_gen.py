import numpy as np

# Table des "prises" (taps) C/A, copiée de votre fichier .m
# Elle va jusqu'au PRN 37
TAP_TABLE = np.array([
    [2, 6], [3, 7], [4, 8], [5, 9], [1, 9], [2, 10], [1, 8], [2, 9], [3, 10],
    [2, 3], [3, 4], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [1, 4], [2, 5],
    [3, 6], [4, 7], [5, 8], [6, 9], [1, 3], [4, 6], [5, 7], [6, 8], [7, 9],
    [8, 10], [1, 6], [2, 7], [3, 8], [4, 9], [5, 10], [4, 10], [1, 7], [2, 8],
    [4, 10]
])

def generate_ca_code(prn_list, fs_ratio=1.0, bipolar=False):
    """
    Traduction Python du fichier cacode.m de votre professeur.
    
    Args:
        prn_list (list ou np.array): Une liste de PRN à générer (ex: [5] ou [6, 12]).
        fs_ratio (float): Le ratio d'échantillonnage (fs_wav / 1.023e6).
        bipolar (bool): Si True, retourne des valeurs -1 et 1. Si False (défaut), 0 et 1.
    
    Returns:
        np.array: Une matrice (ou un vecteur) des codes C/A.
                  Si N PRNs sont demandés, la forme est (N, 1023 * fs_ratio).
    """
    
    # Handle scalar input
    if np.isscalar(prn_list):
        prn_list = np.array([prn_list])
    else:
        prn_list = np.array(prn_list)
    
    # --- Validation des entrées (comme dans le .m) ---
    if fs_ratio < 1:
        raise ValueError("fs_ratio (fs) doit être 1 ou plus.")
    if np.max(prn_list) > 37 or np.min(prn_list) < 1:
        raise ValueError("Les PRN (sv) doivent être entre 1 et 37.")

    # Longueur du code
    L = 1023
    n = 10
    
    # --- Logique LFSR (identique au .m) ---
    
    # G1 LFSR: x^10+x^3+1
    s = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 1])
    g1_reg = np.ones(n, dtype=int)
    
    # G2j LFSR: x^10+x^9+x^8+x^6+x^3+x^2+1
    t = np.array([0, 1, 1, 0, 0, 1, 0, 1, 1, 1])
    g2_reg = np.ones(n, dtype=int)
    
    # Sélection des "prises" (tap) pour les PRN demandés
    # On ajuste les indices (MATLAB 1-basé -> Python 0-basé)
    tap_sel = TAP_TABLE[prn_list - 1]
    
    # Stockage pour le code de 1023 chips
    # np.zeros va créer une matrice (Nb_PRN x 1023)
    g = np.zeros((len(prn_list), L), dtype=int)
    g2_output = np.zeros(len(prn_list), dtype=int)

    # --- Boucle de génération (identique au .m) ---
    for i in range(L):
        # Sortie G2 : mod(sum(q(tap_sel),2),2)
        # C'est un XOR des deux "prises" pour chaque PRN
        for prn_idx in range(len(prn_list)):
            tap_a = tap_sel[prn_idx, 0] - 1
            tap_b = tap_sel[prn_idx, 1] - 1
            g2_output[prn_idx] = g2_reg[tap_a] ^ g2_reg[tap_b]

        # Sortie finale : mod(g1(n)+g2(:,inc),2)
        # C'est (G1[dernier] XOR G2_output)
        g[:, i] = g1_reg[-1] ^ g2_output
        
        # "Clock" G1
        feedback_g1 = np.sum(g1_reg * s) % 2
        g1_reg = np.roll(g1_reg, 1)
        g1_reg[0] = feedback_g1
        
        # "Clock" G2
        feedback_g2 = np.sum(g2_reg * t) % 2
        g2_reg = np.roll(g2_reg, 1)
        g2_reg[0] = feedback_g2

    # --- Upsampling (si demandé) ---
    # --- Upsampling (si demandé) ---
    if fs_ratio == 1.0:
        result = np.squeeze(g)
    else:
        # C'est la traduction de la boucle "zero order hold"
        L_upsampled = int(np.floor(L * fs_ratio))
        
        # Indices pour le ré-échantillonnage
        indices = np.ceil(np.arange(1, L_upsampled + 1) / fs_ratio).astype(int) - 1
        # Assurer que les indices ne dépassent pas L-1
        indices[indices > L-1] = L-1 

        g_upsampled = g[:, indices]
        result = np.squeeze(g_upsampled)

    # Convert to bipolar if requested: 0 -> 1, 1 -> -1
    if bipolar:
        result = 1 - 2 * result
        
    return result