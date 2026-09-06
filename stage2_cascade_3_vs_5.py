"""
STAGE 2 -- binarni 3-vs-5 "rescue" kaskada
=============================================

Bere HOTOVE predikce ze zakladniho `gas_sensor_pipeline.py` (souboru
`submission_final.csv`) -- ten zustava BEZE ZMENY. Pro VSECHNY testovaci
radky, ktere zakladni pipeline predikovala jako tridu 3, spusti samostatny
binarni ensemble model (3 vs 5), a tam, kde je dostatecne jisty, ze jde
spis o tridu 5, predikci PREPISE. Vsechny ostatni predikce (nepredikovane
jako 3) zustavaji netknute.

MOTIVACE (vse pocitano vyhradne z train.csv):
  - Externi vyhodnoceni (report) ukazalo: trida 3 ma nizkou precision
    (0.783), trida 5 nizky recall (0.733) -- 159 z 599 skutecnych petek
    bylo predikovano jako trojka. Trida 3 tedy funguje jako "kos pro
    nejistotu", kam casto spadnou i skutecne petky.
  - Po odstraneni vlivu koncentrace (cnorm = feat/concentration, aby se
    srovnaly ruzne davky) ma jen TGS2602 velkou rozlisovaci silu mezi
    tridou 3 a 5 (median |Cohen's d| = 2.12 pres 32 priznaku tohoto typu,
    n=3417 v prekryvajicim se koncentracnim rozsahu <=500ppm); ostatni 3
    typy senzoru (TGS2600, TGS2610, TGS2620) temer zadnou (0.05-0.21).
  - Tahle rozlisovaci sila TGS2602 navic soustavne KLESA napric
    trenovacimi batchi (batch1-3: |d|~5-10, batch6-8: |d|~1-3), coherentne
    s nezavisle zjistenym faktem, ze TGS2602 ze 4 typu senzoru drifuje
    nejvic (viz docstring zakladniho pipeline, bod 2 a 10).
  - EMPIRICKY OVERENO: binarni model na SAMOTNEM TGS2602 (80 priznaku: 4
    fyzicke kusy x [raw(8)+cnorm(8)] + mean/std pres kusy pro vsech 8
    pozic) dava na leave-one-batch-out OOF LEPSI binary F1 nez model na
    VSECH 431 priznacich zakladniho pipeline -- potvrzuje, ze zbylych
    ~350 priznaku je pro TUHLE KONKRETNI dvojici cisty sum, ktery jen
    redi signal. Kaskada proto pouziva VYHRADNE TGS2602 priznaky.
  - KLICOVE ZJISTENI (PCA analyza): v PC1 (65% vysvetlene variance)
    zustava TRIDA 3 stabilni napric vsemi batchi (~-8), ALE TRIDA 5
    SOUSTAVNE DRIFTUJE smerem k tride 3 (PC1(5): +15.7 -> 9.7 -> 6.2 ->
    ... -> -2.5 mezi batchi 1 a 9) -- TREND VYPADA SATUROVANY kolem batche
    8-9 (batch8=-2.55, batch9=-2.53, temer identicke). To znamena: cely
    pooled trenink (batch 1-9) obsahuje pro tridu 5 hodne JIZ NEPLATNYCH,
    "stareho stavu" priznaku z ranych batchu, ktere kontaminuji odhad
    toho, jak trida 5 vypada TED (a pravdepodobne i na testu/batchi 10).
    OVERENO: trenink VYHRADNE na "usazenych" batchich 7-9 (misto vsech 1-9)
    dava na leave-one-batch-out validaci (drzenych vzdy batch 7, 8 nebo 9)
    F1=0.987-0.989 misto 0.967-0.973 pri treninku na vsech 1-9 -- batch9
    dokonce presne 1.000. Financi model proto TRENUJE JEN NA BATCHICH 7-9.
  - VYZKOUMANO, ale NEPRINESLO zlepseni: zjednodusene reprezentace (prumer
    pres 4 fyzicke kusy, jen top-8 nejdiskriminativnejsich priznaku,
    within-batch percentilovy rank) VSECHNY prohraly s puvodni 80-
    priznakovou reprezentaci pri radne leave-one-batch-out validaci --
    Cohen's d (univariatni statistika) u vsech tri naznacoval zlepseni,
    skutecny klasifikator vzdy ukazal opak. Take LDA/KNN/Perceptron na
    PCA-redukovanych datech byly VSECHNY horsi nez XGBoost na plnych 80
    priznacich. Vahovani vzorku podle "nedavnosti" (linearni i
    exponencialni) na CELEM 1-9 datasetu doslo cca STEJNEHO vysledku jako
    tvrde orezani na 7-9, ne lepsiho. Jeste uzsi okno (jen 8-9) je uz
    PRILIS malo dat (batch9 OOF F1 padne na 0.924) -- 7-9 je "sweet spot".
  - ENSEMBLE: XGBoost+CatBoost (50/50 soft-voting), oba trenovane JEN na
    batchich 7-9, dava OOF F1 na [7,8,9]=0.989-0.992 misto 0.987-0.989 pro
    kterykoliv model samotny -- genuine ensemblovaci zisk (stejny princip
    jako v hlavnim pipeline). LightGBM byl citelne horsi (0.970) a do
    ensemblu se neprida.
  - PO NASAZENI (skutecny test F1, postupne overovano): 88.1 -> 90.237
    (nested-CV stacking oprava v hlavnim pipeline, bez zmeny skore -- byla
    to oprava REPORTOVANI, ne zdroj zlepseni) -> 91.3 PO teto kaskade
    (trenink na batchich 7-9) -- REALNE OVERENE ZLEPSENI, ne jen OOF signal.

BEZPECNOSTNI POJISTKA (viz krok [4]): rescue (3->5) se aplikuje JEN pokud
testovaci radek lezi v koncentracnim rozsahu, kde trida 5 ma v train.csv
vubec nejaky precedens (viz compute_concentration_ceilings v zakladnim
pipeline) -- jinak by kaskada mohla znovu zavest presne ten extrapolacni
problem (predikce trid bez trenovaciho precedentu), ktery zakladni
pipeline uz resi pomoci concentration_prior_penalty.

Spusteni
--------
    python gas_sensor_pipeline.py       # zakladni pipeline, BEZE ZMENY,
                                         # vytvori submission_final.csv
    python stage2_cascade_3_vs_5.py     # tento skript, nacte
                                         # submission_final.csv a vytvori
                                         # submission_final_cascaded.csv

Ocekavana struktura slozky (stejna jako zakladni pipeline):
    gas_sensor_pipeline.py
    stage2_cascade_3_vs_5.py            <- tento soubor
    submission_final.csv                <- vystup gas_sensor_pipeline.py
    inter-uni-datathon-stream-.../
        train.csv
        test.csv
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score, precision_score, recall_score

import gas_sensor_pipeline as P  # ZAKLADNI pipeline -- POUZITA BEZE ZMENY

RANDOM_STATE = 42

CLASS_A = 3   # trida, kterou zakladni pipeline casto "prehlti" (nizka precision)
CLASS_B = 5   # trida, ktera casto konci jako CLASS_A (nizky recall)
FLIP_THRESHOLD = 0.5  # P(CLASS_B) nad kterou se predikce prepise -- viz krok [3]
                       # pro OOF analyzu ruznych prahu; F1 je stabilni a vysoke
                       # pro cely rozsah 0.3-0.8, 0.5 je rozumny neutralni
                       # vychozi bod.
CEILING_MARGIN = 50.0  # stejna konvence jako concentration_prior_penalty
                       # v zakladnim pipeline (bezpecnostni pojistka, krok [4])
RECENT_BATCHES = [7, 8, 9]  # financi model se trenuje JEN na techto batchich
                             # -- viz docstring: trida 5 driftuje smerem k
                             # tride 3 a trend saturoval kolem batche 8-9;
                             # pouziti celeho 1-9 kontaminuje odhad "jak
                             # trida 5 vypada TED" starym, uz neplatnym stavem

MAIN_SUBMISSION_PATH = "submission_final.csv"
OUTPUT_PATH = "submission_final_cascaded.csv"


def get_binary_models(n_estimators_xgb=300, n_estimators_cb=300):
    """Ensemble XGBoost + CatBoost (50/50 soft-voting) -- EMPIRICKY OVERENO
    (viz docstring): dava lepsi OOF F1 na batchich [7,8,9] nez kterykoliv
    model samotny. LightGBM byl v tomtez testu citelne horsi a proto se
    do ensemblu neprida."""
    return {
        "xgboost": xgb.XGBClassifier(
            objective="binary:logistic", n_estimators=n_estimators_xgb, max_depth=5,
            learning_rate=0.05, tree_method="hist", n_jobs=-1, random_state=RANDOM_STATE,
        ),
        "catboost": CatBoostClassifier(
            loss_function="Logloss", iterations=n_estimators_cb, depth=5,
            learning_rate=0.05, thread_count=-1, verbose=False, random_seed=RANDOM_STATE,
        ),
    }


def fit_ensemble_predict_proba(X_tr, y_tr, X_va):
    """Natrenuje ensemble (viz get_binary_models) a vrati prumerne P(CLASS_B)."""
    probs = []
    for m in get_binary_models().values():
        m.fit(X_tr, y_tr)
        probs.append(m.predict_proba(X_va)[:, 1])
    return np.mean(probs, axis=0)


def leave_one_batch_out_binary_ensemble_oof(X, y_bin, batch_arr):
    """Stejny princip jako leave_one_batch_out_oof() v zakladnim pipeline,
    jen pro binarni ulohu a ensemble 2 modelu (50/50 prumer) -- vraci OOF
    P(CLASS_B) pro kazdy radek."""
    oof = np.full(len(y_bin), np.nan)
    for b in np.unique(batch_arr):
        va = batch_arr == b
        tr = ~va
        if va.sum() == 0 or len(np.unique(y_bin[tr])) < 2:
            continue
        oof[va] = fit_ensemble_predict_proba(X[tr], y_bin[tr], X[va])
    return oof


def main():
    print("=" * 78)
    print("STAGE 2: binarni 3-vs-5 rescue kaskada")
    print("(zakladni gas_sensor_pipeline.py zustava BEZE ZMENY)")
    print("=" * 78)

    print("\n[0] Nacitani dat + hlavni pipeline predikci...")
    train_df, test_df = P.load_data()
    feat_cols = [f"feat_{i}" for i in range(1, 129)]
    main_sub = pd.read_csv(MAIN_SUBMISSION_PATH)
    n_pred_a = int((main_sub["gas_class"] == CLASS_A).sum())
    print(f"    train: {train_df.shape}, test: {test_df.shape}")
    print(f"    hlavni submission ('{MAIN_SUBMISSION_PATH}'): {main_sub.shape}, "
          f"predikovano jako trida {CLASS_A}: {n_pred_a}")

    y_full = train_df["gas_class"].to_numpy(dtype=int)
    batch_arr = train_df["batch"].to_numpy(dtype=int)
    mask_ab = np.isin(y_full, [CLASS_A, CLASS_B])
    y_bin = (y_full[mask_ab] == CLASS_B).astype(int)
    batch_ab = batch_arr[mask_ab]
    print(f"    trenovaci radky trida {CLASS_A}: {(y_bin == 0).sum()}, "
          f"trida {CLASS_B}: {(y_bin == 1).sum()}")

    print("\n[1] Feature engineering -- TGS2602-specificke priznaky "
          "(sdilene s hlavnim pipeline pres build_per_type_features)...")
    per_type_train = P.build_per_type_features(train_df, feat_cols)
    per_type_test = P.build_per_type_features(test_df, feat_cols)
    X_tgs_train = per_type_train["TGS2602"][mask_ab]
    X_tgs_test = per_type_test["TGS2602"]
    print(f"    dim = {X_tgs_train.shape[1]} (jen TGS2602 -- viz docstring, proc "
          f"ne plna 431-priznakova sada)")

    print("\n[2] Validace (leave-one-batch-out OOF, ensemble XGBoost+CatBoost)...")
    oof_tgs = leave_one_batch_out_binary_ensemble_oof(X_tgs_train, y_bin, batch_ab)
    valid = ~np.isnan(oof_tgs)
    f1_tgs = f1_score(y_bin[valid], (oof_tgs[valid] > 0.5).astype(int))
    print(f"    OOF binary F1 (vsech 9 batchu) = {f1_tgs:.4f}")

    print(f"\n[3] Volba prahu pro P(trida {CLASS_B}) na OOF datech...")
    for thr in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        pred = (oof_tgs[valid] > thr).astype(int)
        prec = precision_score(y_bin[valid], pred, pos_label=1, zero_division=0)
        rec = recall_score(y_bin[valid], pred, pos_label=1, zero_division=0)
        f1 = f1_score(y_bin[valid], pred, zero_division=0)
        print(f"      prah={thr}: precision({CLASS_B})={prec:.4f}  "
              f"recall({CLASS_B})={rec:.4f}  F1={f1:.4f}")
    print(f"    -> Pouzivam FLIP_THRESHOLD={FLIP_THRESHOLD} (viz komentar u konstanty).")

    print(f"\n[3b] Overeni: trenink JEN na 'usazenych' batchich {RECENT_BATCHES} vs "
          f"vsech 1-9 (viz docstring -- trida 5 driftuje k tride 3, trend saturoval)...")
    f1s_all, f1s_recent = [], []
    for b in RECENT_BATCHES:
        va_b = batch_ab == b
        tr_all = ~va_b
        tr_recent = np.isin(batch_ab, RECENT_BATCHES) & ~va_b
        if va_b.sum() < 5 or len(np.unique(y_bin[tr_recent])) < 2:
            continue
        p_all = fit_ensemble_predict_proba(X_tgs_train[tr_all], y_bin[tr_all], X_tgs_train[va_b])
        p_rec = fit_ensemble_predict_proba(X_tgs_train[tr_recent], y_bin[tr_recent], X_tgs_train[va_b])
        f1s_all.append(f1_score(y_bin[va_b], (p_all > 0.5).astype(int), zero_division=0))
        f1s_recent.append(f1_score(y_bin[va_b], (p_rec > 0.5).astype(int), zero_division=0))
    print(f"    Trenink na VSECH batchich 1-9: F1 na {RECENT_BATCHES} = "
          f"{[f'{x:.4f}' for x in f1s_all]}  prumer={np.mean(f1s_all):.4f}")
    print(f"    Trenink JEN na batchich {RECENT_BATCHES}:    F1 na {RECENT_BATCHES} = "
          f"{[f'{x:.4f}' for x in f1s_recent]}  prumer={np.mean(f1s_recent):.4f}")
    print(f"    -> Financi model bude trenovan JEN na batchich {RECENT_BATCHES}.")

    print("\n[4] Trenink financiho binarniho ENSEMBLU (XGBoost+CatBoost, 50/50) -- "
          f"JEN na 'usazenych' batchich {RECENT_BATCHES}...")
    recent_mask = np.isin(batch_ab, RECENT_BATCHES)
    X_final_train = X_tgs_train[recent_mask]
    y_final_train = y_bin[recent_mask]
    print(f"    Trenovacich radku po omezeni na {RECENT_BATCHES}: {len(y_final_train)} "
          f"(z puvodnich {mask_ab.sum()})")
    test_proba_b = fit_ensemble_predict_proba(X_final_train, y_final_train, X_tgs_test)

    print("\n[5] Bezpecnostni pojistka: koncentracni strop pro tridu "
          f"{CLASS_B} (pocitano VYHRADNE z train.csv, stejna metoda jako "
          "concentration_prior_penalty v zakladnim pipeline)...")
    ceilings = P.compute_concentration_ceilings(train_df)
    ceiling_b = float(ceilings[CLASS_B - 1])
    test_concentration = test_df["concentration"].to_numpy(dtype=np.float64)
    plausible_for_b = test_concentration <= (ceiling_b + CEILING_MARGIN)
    print(f"    Strop tridy {CLASS_B}: {ceiling_b}ppm (+ margin {CEILING_MARGIN}ppm)")
    print(f"    Testovacich radku, kde je trida {CLASS_B} vubec fyzikalne prijatelna: "
          f"{plausible_for_b.sum()}/{len(test_df)}")

    print("\n[6] Aplikace kaskady na radky predikovane hlavnim pipeline jako "
          f"trida {CLASS_A}...")
    sub = main_sub.copy()
    is_pred_a = (sub["gas_class"] == CLASS_A).to_numpy()
    should_flip = is_pred_a & (test_proba_b > FLIP_THRESHOLD) & plausible_for_b
    blocked_by_ceiling = is_pred_a & (test_proba_b > FLIP_THRESHOLD) & (~plausible_for_b)

    n_flipped = int(should_flip.sum())
    n_blocked = int(blocked_by_ceiling.sum())
    print(f"    Prepsano {CLASS_A}->{CLASS_B}: {n_flipped} radku "
          f"(z {n_pred_a} puvodne predikovanych jako {CLASS_A})")
    if n_blocked > 0:
        print(f"    POJISTKA zabranila prepsani u {n_blocked} radku "
              f"(binarni model byl jisty, ale koncentrace je mimo strop tridy "
              f"{CLASS_B} -- pravdepodobne spravne, ze zustava {CLASS_A}).")

    sub.loc[should_flip, "gas_class"] = CLASS_B
    sub.to_csv(OUTPUT_PATH, index=False)

    print(f"\nHotovo. Kaskadovana submission ulozena do '{OUTPUT_PATH}'.")
    print("Distribuce trid PRED kaskadou:")
    print(main_sub["gas_class"].value_counts().sort_index())
    print("Distribuce trid PO kaskade:")
    print(sub["gas_class"].value_counts().sort_index())


if __name__ == "__main__":
    main()
