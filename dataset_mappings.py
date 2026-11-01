BCSS_TO_UNIFIED = {
    "tumor": "Tumor",
    "dcis": "Tumor",
    "fat": "Fat",
    "stroma": "Stroma",
    "lymphocytic_infiltrate": "Inflammatory",
    "plasma_cells": "Inflammatory",
    "other_immune_infiltrate": "Inflammatory",
    "necrosis_or_debris": "Necrosis",
    "nerve": "Nerve",
    "blood": "Vascular",
    "blood_vessel": "Vascular",
    "skin_adnexa": "Skin adnexa",
    "normal_acinus_or_duct": "Epithelial",
    "mucoid_material": "Mucin",
    "metaplasia_NOS": "Epithelial",
    "exclude": "Background"
}

SPIDER_SKIN_TO_UNIFIED = {
    # --- Tumor / malignant lesions ---
    "Basal cell carcinoma": "Tumor",
    "Squamous cell carcinoma": "Tumor",
    "Carcinoma in situ": "Tumor",
    "Invasive melanoma": "Tumor",
    "Melanoma in situ": "Tumor",
    "Mercel cell carcinoma": "Tumor",
    "Kaposi’s sarcoma": "Tumor",

    # --- Fat / adipose ---
    "Fat": "Fat",

    # --- Stroma / connective / muscle ---
    "Collagen": "Stroma",
    "Muscle": "Stroma",

    # --- Inflammatory / immune infiltrates ---
    "Inflammation": "Inflammatory",

    # --- Necrosis / debris ---
    "Necrosis": "Necrosis",

    # --- Nerve structures ---
    "Nerves": "Nerve",

    # --- Vascular / blood structures ---
    "Vessels": "Vascular",

    # --- Skin adnexa (appendages) ---
    "Follicle": "Skin adnexa",
    "Sebaceous gland": "Skin adnexa",
    "Apocrine glands": "Skin adnexa",

    # --- Epithelial / surface tissue ---
    "Epidermis": "Epithelial",
    "Keratin": "Epithelial",
    "Solar elastosis": "Stroma",
    "Actinic keratosis": "Epithelial",
    "Seborrheic keratosis": "Epithelial",
    "Nevus": "Epithelial",
    "Wart": "Epithelial",
}

SPIDER_THORAX_TO_UNIFIED = {
    # Parenchyma / linings / glands
    "Alveoli": "Epithelial",            # pneumocytes/alveolar lining
    "Bronchial glands": "Epithelial",   # seroinous glands
    "Pleura": "Epithelial",             # mesothelium as lining epithelium 

    # Stroma / connective tissuesmuc
    "Bronchial cartilage": "Stroma",
    "Fibrosis": "Stroma",

    # Inflammation / lymphoid
    "Chronic inflammation + fibrosis": "Inflammatory",
    "Lymph node": "Inflammatory",

    # Vascular & blood products
    "Vessel": "Vascular",
    "Hemorrhage": "Vascular",           # extravasated blood grouped with vascular/blood

    # Non-viable material
    "Detritus": "Necrosis",             # acellular debris/necrotic material

    # Tumor
    "Tumor non-small cell": "Tumor",
    "Tumor small cell": "Tumor",
    "Tumor soft": "Tumor",              # keep as malignant unless dataset says otherwise
}

SPIDER_COLORECTAL_TO_UNIFIED = {
    # Malignant epithelium
    "Adenocarcinoma high grade": "Tumor",
    "Adenocarcinoma low grade":  "Tumor",

    # Precursor / benign epithelial lesions
    "Adenoma high grade":        "Epithelial",
    "Adenoma low grade":         "Epithelial",
    "Hyperplastic polyp":        "Epithelial",
    "Sessile serrated lesion":   "Epithelial",

    # Fat / soft tissue
    "Fat":                       "Fat",

    # Stroma / connective / muscle / extracellular
    "Stroma healthy":            "Stroma",
    "Muscle":                    "Stroma",
    "Mucus":                     "Mucin",

    # Inflammation / immune
    "Inflammation":              "Inflammatory",

    # Non-viable / debris
    "Necrosis":                  "Necrosis",

    # Vessels / blood
    "Vessels":                   "Vascular",
}


SPIDER_BREAST_TO_UNIFIED = {
    # --- Malignant epithelium ---
    "Invasive non-special type carcinoma": "Tumor",
    "Lobular invasive carcinoma":          "Tumor",

    # --- In situ carcinoma ---
    "Ductal carcinoma in situ (high-grade)": "Tumor",
    "Ductal carcinoma in situ (low-grade)":  "Tumor",

    # --- Malignant biphasic tumor ---
    "Malignant phyllodes tumor":            "Tumor",

    # --- Benign biphasic tumor ---
    "Benign phyllodes tumor":               "Stroma",
    # Rationale: phyllodes tumors are stromal neoplasms with epithelial clefts; 
    # BCSS has nothing equivalent, but the stroma mapping matches your taxonomy design.

    # --- Benign epithelial proliferations / normal epithelium ---
    "Typical ductal hyperplasia":           "Epithelial",
    "Adenosis":                             "Epithelial",
    "Sclerosing adenosis":                  "Epithelial",
    "Normal ducts":                         "Epithelial",
    "Normal lobules":                       "Epithelial",

    # --- Fat / adipose ---
    "Fat":                                  "Fat",

    # --- Stromal / connective tissue ---
    "Fibrosis":                             "Stroma",
    "Fibroadenoma":                         "Stroma",
    # Rationale: fibroadenoma is a stromal lesion with epithelial components, 
    # but your system groups similar things under “Stroma”.

    "Fibrocystic changes":                  "Stroma",
    # Rationale: mixture of cysts, fibrosis, apocrine metaplasia → best fit = Stroma.

    # --- Inflammatory / reactive ---
    "Lipogranuloma":                        "Inflammatory",
    # Rationale: granulomatous inflammation involving fat.

    # --- Non-viable tissue ---
    "Necrosis":                             "Necrosis",

    # --- Vascular ---
    "Vessels":                              "Vascular",
}

LUB_LUNG_TO_UNIFIED = {
    "01_TUMOR":     "Tumor",
    "02_STROMA":    "Stroma",
    "04_IMMUNE":    "Inflammatory",
    "05_VESSEL":    "Vascular",
    "06_BRONCHI":   "Epithelial",
    "07_NECROSIS":  "Necrosis",
    "08_LUNG":      "Epithelial",  # alveolar epithelium / lung parenchyma
    "09_BACKGROUND":"Background",
}

KATHER_TO_UNIFIED = {
    "ADI":  "Fat",               # Adipose tissue
    "BACK": "Background",        # Non-tissue
    "DEB":  "Necrosis",          # Debris / necrotic material
    "LYM":  "Inflammatory",      # Lymphocytes
    "MUC":  "Mucin",             # Mucin
    "MUS":  "Stroma",            # Smooth muscle (grouped with stroma as before)
    "NORM": "Epithelial",        # Normal colon mucosa
    "STR":  "Stroma",            # Cancer-associated stroma
    "TUM":  "Tumor",             # Adenocarcinoma epithelium
}

KATHER_5000_TO_UNIFIED = {
    "01_TUMOR":     "Tumor",               # Invasive carcinoma epithelium
    "02_STROMA":    "Stroma",              # Fibrous / desmoplastic stroma
    "04_LYMPHO":    "Inflammatory",         # Lymphocyte-rich tissue
    "05_DEBRIS":    "Necrosis",             # Debris / necrotic material
    "06_MUCOSA":    "Epithelial",           # Normal colon mucosa
    "07_ADIPOSE":   "Fat",                  # Adipose tissue
    "08_EMPTY":     "Background",           # Non-tissue / whitespace
}

PROSTATE_TO_UNIFIED = {
    "gland": "Epithelial",
    "tumor": "Tumor"
}

