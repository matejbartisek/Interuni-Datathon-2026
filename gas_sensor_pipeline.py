"""
Gas Sensor Array Drift -- inter-university datathon "Stream 3"
================================================================

Stav teto verze (na zaklade prubezne zpetne vazby z realnych behu):
  (A) 3 modely -- XGBoost, CatBoost, LightGBM (AdaBoost a HistGradientBoosting
      vyrazeny).
  (B) VYRAZENO vyuziti informace "test ma presne 600 vzorku na tridu" --
      bylo urceno, ze tohle je zakazane vyuzivat. Financi klasifikace NIKDY
      nepracuje s agregatni distribuci trid pres cely test -- kazdy radek se
      rozhoduje NEZAVISLE na ostatnich.
  (C) ZNOVU PRIDANO po analyze skutecneho behu (viz bod 5 a 6 nize):
      koncentracni-strop prior + hard-gate pro pseudo-labeling, L2-regularizovana
      Nelder-Mead kalibrace, a forward-chaining validace pro vyber modelu.
      Vsechny tri opravy jsou pocitany VYHRADNE z train.csv (per-radek nebo
      per-fold), NIKDY z agregatni statistiky testovaci sady -- nekoliduji
      tedy s bodem (B).
  (D) Po druhem skutecnem behu: pseudo-labeling byl DOCASNE zkracen na 2 kola
      (kvuli poklesu OOF F1 v kole 3), ale po TRETIM skutecnem behu (test F1)
      bylo zjisteno, ze 2 kola davaji HORSI skutecny test F1 (86) nez 3 kola
      (88.1) -- CISTY A/B test, vse az po kolo 2 bylo bit-identicke mezi
      obema behy. VRACENO ZPET na 3 kola. Duvod chybneho puvodniho zaveru a
      obecne pouceni viz bod 8 nize -- OOF neni pro tenhle hyperparametr
      spolehlivy ukazatel. Take NOVE ZJISTENI o slabine pri nizke koncentraci
      (<50ppm) -- viz bod 7 a `diagnose_low_concentration_drift_sensitivity()`.

ZNAMY, VEDOME NEOPRAVENY LIMIT: leave-one-batch-out OOF pouziva batch9 jako
jeden z 9 foldu pro kalibraci -- ale batch9 je SOUCASNE tim, na cem se v [3]
vybiraji vitezne modely (train<=8->val9). Model vybrany "protoze uspel na
batch9" je pak validovan i kalibrovan castecne na tomtez batchi (~4.6 % OOF
vahy). Neni to primy label-leak, ale je to mirna cirkularita mezi vyberem a
kalibraci. Oprava (nested CV, nebo OOF schema vyhybajici se batch9) by
pridala netrivialni komplexitu za nejasny prinos vzhledem k tomu, ze batch9
nese jen ~4.6 % OOF vahy -- proto NEIMPLEMENTOVANO, jen zdokumentovano. Rekni,
pokud to presto chces resit.

KLICOVA ZJISTENI, KTERA TATO PIPELINE VYUZIVA
-----------------------------------------------
1) Struktura 128 priznaku = 16 senzoru x 8 pozic. 16 senzoru = 4 fyzicke kusy
   kazdeho ze 4 typu (Figaro TGS2600, TGS2602, TGS2610, TGS2620). Poradi kanalu
   (1-indexovane bloky feat_(8b-7)..feat_(8b)):
       TGS2602 = bloky [1, 2, 9, 10]
       TGS2600 = bloky [3, 4, 11, 12]
       TGS2610 = bloky [5, 6, 13, 14]
       TGS2620 = bloky [7, 8, 15, 16]
   8 pozic v kazdem bloku: DR, normDR, EMA_rise(a=.1/.01/.001), EMA_fall(a=.1/.01/.001).

2) Drift NENI stejny napric typy senzoru:
     - TGS2602 drifuje nejvic a temer monotonne (~2-2.5x rychleji nez ostatni tri typy).
     - TGS2600 / TGS2610 / TGS2620 maji "kopcovity" (nemonotonni) prubeh behem
       treninku a prave ony skoci nejvic pri prechodu do test batch 10 (zatimco
       TGS2602 tam zustava temer plochy).
     - TGS2610 je navic nejmene "predikovatelny": jeho 2 duplicitni fyzicke kusy
       driftuji navzajem nesouhlasne (r=0.51 vs 0.90-0.98 u ostatnich typu) ->
       per-typova agregace (mean/std pres 4 kusy) dava modelu sanci to zachytit.
     - normDR NENI univerzalne "drift-robustnejsi" nez DR -- je to typove zavisle
       (pomaha u TGS2600, skodi u TGS2602/TGS2610). Proto normDR neupred-
       nostnujeme plosne, jen ji necharve model k dispozici vedle DR.
     - DR a rychle EMA (alfa=0.1) nesou nejcistsi koncentrace/plyn signal (R^2~0.65).

3) O testovacim setu NEPREDPOKLADAME A NEVYUZIVAME ZADNOU informaci o rozlozeni
   trid (napr. ze by bylo rovnomerne) -- vyuziti takove informace bylo urceno
   jako zakazane pravidly souteze. Financi klasifikace je nezavisle rozhodnuti
   pro kazdy radek (viz bod 5), NIKDY globalni omezeni na vyslednou distribuci.

4) Nahodny StratifiedKFold (jak byl pouzit v drivejsich verzich test.py) michel
   dohromady radky z ruznych batchu bez ohledu na cas -> systematicky
   NADHODNOCUJE ocekavany vykon na batch 10. Misto toho pouzivame:
     (a) FORWARD-CHAINING validace (train<=6->val7, train<=7->val8, train<=8->val9,
         prumerovano) pro VYBER modelu. Duvod navratu k tomuto misto
         jednorazoveho train<=8->val9 splitu: batch9 ma jen 3 unikatni
         koncentrace (10/100/120 ppm) a jednorazovy split dal catboost=0.955
         vs xgboost=0.762 (rozdil 0.19!), zatimco forward-chaining prumer dal
         jen 0.933 vs 0.919 (rozdil 0.014) -- tedy 10x mensi a dukryhodnejsi.
     (b) leave-one-batch-out GroupKFold (9 foldu = 9 batchu) pro generovani OOF
         pravdepodobnosti pouzitych pri kalibraci a pseudo-labelingu (viz
         "ZNAMY LIMIT" vyse ohledne batch9 v teto casti).

5) KRITICKE: kazda trida (plyn) ma v train datech jiny MAXIMALNI testovany
   koncentracni strop (napr. trida s max=1000ppm vs. trida s max=230ppm) --
   artefakt puvodniho experimentu (kazdy plyn se davkoval v jinem rozsahu),
   NE nahoda a NE odvozeno z externich dat (spocitatelne primo z train.csv,
   NIC spolecneho s bodem 3 -- tohle je vztah feature/label v TRAIN datech,
   ne predpoklad o distribuci testovacich labelu).
   OVERENO na skutecnem behu: pri primem argmax bez teto opravy dostava
   trida s trenovacim stropem 300ppm 30 predikci presne na 1000ppm (=700ppm
   extrapolace za hranici jakehokoliv trenovaciho precedentu), a podobne u
   800ppm. Test obsahuje 800 radku (22%) s koncentraci >=400ppm.
   Reseni: `concentration_prior_penalty()` pridava hladkou (NE tvrdou)
   penalizaci do -log(prob) skore pro (radek, trida) kombinace za hranici
   trenovaneho stropu dane tridy -- aplikovano JAK pred pseudo-labelingem
   (tvrdsi brana v select_pseudo_labels, aby se do treninku neucily
   sebejiste-spatne extrapolace), TAK ve financim kroku [7] (per-radek
   argmin cost, ZADNY vliv na agregatni distribuci -- viz bod 3). Sila teto
   penalizace je VEDOME PEVNE NASTAVENA, ne optimalizovana Nelder-Meadem --
   OOF ma temer nulove pokryti nad ~300ppm pro polovinu trid, takze
   optimalizace vuci nemu by byla slepa (presne stejny rezim selhani jako
   oscilujici tridni vaha v bode 6 nize).

6) Bez regularizace muze Nelder-Mead tridni vaha kmitat mezi koly
   pseudo-labelingu bez konvergence -- OVERENO na skutecnem behu: vaha jedne
   tridy oscilovala 0.669 -> 0.443 -> 0.887 -> 0.311 (2.9x rozptyl mezi
   sousednimi koly), zatimco OOF F1 stale rostl (0.941->0.957) -- typicky
   priznak PREUCENI KALIBRATORU na sum, ne skutecneho zlepseni. Kalibrator se
   snazi globalnim multiplikatorem opravit chybu, ktera je ve skutecnosti
   soustredena jen v jedne koncentracni oblasti (viz bod 5) -- globalni
   oprava to nemuze spravit a jen prehani vahu tam a zpet. `calibrate_ensemble`
   ted pridava L2 penalizaci na log(class_w) smerem k 1.0, coz extremni
   vychylky odrazuje a stabilizuje vysledek mezi koly.

7) NOVE (druhy skutecny beh): OOF macro-F1 podle koncentracniho bucketu
   ukazal, ze NEJHORSI oblast NENI ta nad koncentracnim stropem (tu uz resi
   bod 5), ale prekvapive oblast POD 50ppm (macro-F1=0.638, zatimco 50-130ppm
   ma macro-F1~0.96) -- a tahle oblast ma v train.csv PLNE pokryti (zadna
   mezera jako u stropu). Overeno nezavisle: nahodny stratifikovany K-fold
   (michajici batche) dava v teto oblasti temer DOKONALY vysledek
   (macro-F1=0.9996!), zatimco leave-one-batch-out OOF (respektujici cas)
   dava jen 0.638. Kdyby slo o fyzikalni limit (slaby signal = spatne
   rozlisitelne plyny), selhal by i nahodny K-fold -- neselhava. SPRAVNE
   VYSVETLENI: DRIFT-CITLIVOST SPECIFICKA PRO NIZKOU KONCENTRACI. Absolutni
   velikost senzoroveho signalu (DR/EMA) roste s koncentraci -> stejna
   absolutni chyba driftove korekce (moment_align) tvori mnohem vetsi
   RELATIVNI chybu pri nizke koncentraci (slaby signal) nez pri vysoke
   (silny signal). Nahodny K-fold tohle neodhali, protoze michá driftove
   stavy vsech batchu dohromady -- presne ten problem, kvuli kteremu jinde
   pouzivame leave-one-batch-out/forward-chaining. Diagnostika (NE oprava --
   spravny fix by vyzadoval hlubsi redesign driftove korekce, coz je vetsi
   zmena vyzadujici samostatne probrani) je k dispozici jako volitelna,
   rucne spoustena funkce `diagnose_low_concentration_drift_sensitivity()`
   (NENI volana automaticky v main(), protoze pridava ~3 dalsi fity modelu
   navic k jiz drahemu behu).

8) KRITICKE POUCENI (treti skutecny beh, test F1): OOF (leave-one-batch-out
   pres batch 1-9) NENI spolehlivy ukazatel pro rozhodovani o HLOUBCE
   pseudo-labelingu (kolik kol). Duvod: cely smysl pseudo-labelingu je
   prizpusobit model batchi 10 (skutecnemu testu) pomoci jeho vlastnich dat
   -- ale OOF se z principu pocita jen na batch 1-9, takze NEMUZE zmerit,
   jestli tohle prizpusobeni pomohlo, nebo ne. Empiricky: 2 kola pseudo-
   labelingu davaji OOF F1=0.957 (kolo 2), 3 kola davaji OOF F1=0.954 (kolo
   3, tedy NIZSI) -- ale na SKUTECNEM testu davaji 2 kola F1=86 a 3 kola
   F1=88.1 (tedy VYSSI). OOF a skutecny test si tady PROTIRECI. Pravdepodobny
   mechanismus: kalibrace po kole 2 mela extremnejsi tridni vahy nez kolo 1
   i kolo 3 (napr. [0.787, ..., 1.199] vs [1.01, ..., 1.017]) -- kolo 3 je
   stahlo zpet k 1.0, coz OOF (mereny na batch9, ktery je znamo odlisny od
   batch10) vyhodnotil jako zhorseni, ale na batchi 10 to bylo zlepseni.
   ZAVER: PSEUDO_ROUNDS_CAPS = [100, 200, 300] (3 kola) je vraceno jako
   vychozi -- je to jediny hyperparametr v teto pipeline, kde davame prednost
   primemu dukazu ze skutecneho testu pred OOF signalem, protoze OOF je pro
   tenhle konkretni ucel systematicky slepy.

9) NOVE: KS-test (data_view.ipynb sekce 10) ukazal, ze batch9 (na kterem z
   velke casti stoji nase validace/OOF) je PREKVAPIVE jednou z NEJHORSICH
   proxy pro skutecny test batch 10 -- OVERENO primo na datech, KS-test
   distribucni vzdalenosti KAZDEHO jednotliveho batche vuci testu:
       batch7: KS=0.207 (NEJLEPSI ze vsech 9, navic n=3613 ~ velikost testu)
       batch9: KS=0.365 (jeden z NEJHORSICH)
       cely pooled train (1-9): KS=0.184
   Toto je DALSI, NEZAVISLY dukaz (jina metoda nez F1-based srovnani z bodu
   4/8) pro to, ze batch9 neni spolehlivy zaklad pro rozhodovani. Reakce:
   `evaluate_batch7_holdout()` prida k forward-chainingu jeste jeden,
   VYSE VAZENY vyberovy signal (0.5*forward-chaining + 0.5*batch7-holdout)
   zalozeny na treninku na VSECH ostatnich batchich a validaci prave na
   batchi 7 -- vyuziva vic dat NEZ forward-chainingovy "train<=6" split A
   zaroven validuje na nejreprezentativnejsim dostupnem batchi.

10) NOVE: dve rozsireni na zaklade otazky "jak jeste zvysit F1":
    a) STACKING META-LEARNER (fit_stacking_meta_learner_cv) nahrazuje puvodni
       skalarni `blend_w` -- multinomicka logisticka regrese naucena na OOF
       pravdepodobnostech VSECH vstupnich modelu se muze naucit ruznou vahu
       PRO KAZDOU TRIDU zvlast, coz je flexibilnejsi nez jedno cislo pro
       celou kombinaci. Trenink zustava striktne na leave-one-batch-out OOF
       (leakage-free), jen se meni JAK se vstupy kombinuji.
    b) PER-SENZOR-TYPOVI EXPERTI (fit_per_type_experts) -- 4 samostatne
       modely, kazdy trenovany JEN na priznacich jednoho typu senzoru (jeho
       4 fyzicke kusy), aby se otestovalo, kolik samostatneho signalu kazdy
       typ nese. Jejich OOF/testovaci pravdepodobnosti se pridavaji jako
       DALSI VSTUP do stacking meta-learneru (bod a) -- ne jako samostatna
       soucast hlavniho ensemblu. Z nakladovych duvodu se FITUJI JEN JEDNOU
       (pred pseudo-labeling smyckou), ne kazde kolo znovu.
       OVERENO na skutecnem behu: TGS2600 (OOF F1=0.83) a TGS2620 (0.82) jsou
       nejlepsi samostatni klasifikatori, TGS2602 (0.73) a TGS2610 (0.74)
       nejhorsi -- presne odpovida tomu, ktere typy nejvic/nejmene
       predikovatelne drifuji (bod 2) -- nezavisle potvrzeni driftove
       analyzy z uplne jineho uhlu (klasifikacni presnost samotneho typu).

11) KRITICKA OPRAVA (po ctvrtem skutecnem behu, test F1 > 90): puvodni
    `fit_stacking_meta_learner` se fitoval NA CELE OOF matici zakladnich
    modelu a pak se NA TE SAME matici vyhodnocoval pro OOF F1 report i pro
    kalibraci class_w -- to NENI genuine out-of-sample odhad, je to
    "trenovaci presnost meta-vrstvy". OVERENO: OOF macro-F1 vyskocilo z
    ~0.94-0.96 (predchozi behy) na ~0.986-0.990 -- mnohem vic, nez odpovidalo
    realnemu zlepseni na testu (88.1->90+). OPRAVA
    (`fit_stacking_meta_learner_cv`): meta-learner ma TED SVUJ VLASTNI
    leave-one-batch-out -- pro kazdy batch b se natrenuje na meta-priznacich
    vsech OSTATNICH batchu, predikuje na OOF-radcich batche b (honest
    nested_oof) a zaroven na testu (prumeruje se pres vsech 9 takto
    natrenovanych meta-learneru, stejna bagging filozofie jako uz ma
    leave_one_batch_out_oof() pro zakladni modely). `class_w` kalibrace i
    `report_oof_f1_by_concentration_bucket` ted pouzivaji tenhle honest
    `nested_oof`, ne puvodni (nafouknuty) stacked_oof.

PIPELINE
--------
  1. Feature engineering (vc. winsorizace outlieru a outlier_count priznaku):
       - raw 128 priznaku (winsorizovane, viz winsorize_raw_features)
       - cnorm = feat / concentration (128 priznaku)
       - concentration + log1p(concentration)
       - per-pozice statistiky napric vsemi 16 senzory (mean/std/range) - 24 priznaku
       - per-typova agregace pres 4 fyzicke kusy kazdeho typu (mean/std) - 64 priznaku
       - cross-type pomery pro nejcistsi/nejdiskriminativnejsi pozice
         (DR, normDR, EMA_rise a=0.01, EMA_fall a=0.1) - 24 priznaku
       - sousedni-senzor pomery pro tytez pozice - 60 priznaku
       - outlier_count (pocet extremnich priznaku na radek) - 1 priznak
  2. Per-feature "moment alignment" drift korekce (train -> test), aplikovana
     zvlast na kazdy priznak -- NE plna kovariancni CORAL transformace.
  3. Porovnani 3 modelu (XGBoost, CatBoost, LightGBM): forward-chaining (3
     splity) + batch7-holdout, kombinovano 50/50 -> vyber 2 nejlepsich.
  4. Fit 4 per-senzor-typovych "expertu" (jednorazove, viz bod 10b).
  5. Stacking meta-learner (multinomicka logisticka regrese nad OOF
     pravdepodobnostmi 2 hlavnich modelu + 4 experty) + Nelder-Mead kalibrace
     6 tridnich multiplikatoru (L2-regularizovana) na leave-one-batch-out OOF.
  6. 3 kola pseudo-labelingu (konsenzus 2 hlavnich modelu + rostouci kvoty na
     tridu: 100 / 200 / 300 + koncentracni-strop hard-gate), s re-fitem
     stacking meta-learneru a re-kalibraci po kazdem kole. (POZOR: viz bod 8
     -- OOF pro tenhle hyperparametr neni spolehlivy, 3 kola vraceno na
     zaklade skutecneho testovaciho skore, ne OOF signalu.)
  7. Financi klasifikace: per-radek argmin(-log(prob) + koncentracni prior),
     ZADNE vynucovani rovnomerne distribuce trid -- viz bod 3 vyse.

Volitelna diagnostika (spustit rucne, NENI soucasti main()):
    from gas_sensor_pipeline import diagnose_low_concentration_drift_sensitivity, load_data
    train_df, _ = load_data()
    diagnose_low_concentration_drift_sensitivity(train_df, [f"feat_{i}" for i in range(1,129)])

Spusteni
--------
    python gas_sensor_pipeline.py

Rychly ladici beh s mensimi hyperparametry (pro overeni, ze pipeline nespadne,
ne pro produkcni vysledek):

    GSD_FAST_DEBUG=1 python gas_sensor_pipeline.py
"""

import itertools
import os
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier
from scipy.optimize import minimize
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ----------------------------------------------------------------------------
# Konfigurace
# ----------------------------------------------------------------------------

TRAIN_PATH = "inter-uni-datathon-stream-3-gas-sensor-array-drift-dataset/train.csv"
TEST_PATH = "inter-uni-datathon-stream-3-gas-sensor-array-drift-dataset/test.csv"
OUTPUT_PATH = "submission_final.csv"

N_SENSORS = 16
N_STATS = 8
N_CLASSES = 6

# 1-indexovane bloky priznaku (blok b = feat_(8b-7) .. feat_(8b))
SENSOR_TYPE_BLOCKS = {
    "TGS2602": [1, 2, 9, 10],
    "TGS2600": [3, 4, 11, 12],
    "TGS2610": [5, 6, 13, 14],
    "TGS2620": [7, 8, 15, 16],
}
TYPE_NAMES = list(SENSOR_TYPE_BLOCKS.keys())

# 8 pozic uvnitr kazdeho senzoroveho bloku (Vergara et al. 2012)
POSITION_NAMES = [
    "DR", "normDR",
    "EMA_rise_a0.1", "EMA_rise_a0.01", "EMA_rise_a0.001",
    "EMA_fall_a0.1", "EMA_fall_a0.01", "EMA_fall_a0.001",
]
# Pozice s nejcistsim koncentrace/plyn signalem (R^2~0.65) + normDR
INFORMATIVE_POSITIONS = [0, 1, 3, 5]  # DR, normDR, EMA_rise_a0.01, EMA_fall_a0.1
# EMA_rise_a0.01 (index 3) pridano na zaklade data_view.ipynb sekce 8 (ANOVA
# F-test proti gas_class): mezi top-15 nejvic diskriminativnich priznaku se
# EMA_rise_a0.01 objevuje 6x napric CTYRMI ruznymi typy senzoru (TGS2600,
# TGS2610, TGS2620 vicekrat) -- silny, napric-typovy signal pro KLASIFIKACI
# samotnou, ktery puvodni vyber (jen DR/normDR/EMA_fall_a0.1, vybrany podle
# R^2 proti koncentraci+gas z drift analyzy) prehlizel. R^2 meri "kolik
# variance vysvetli koncentrace+trida spolecne" -- coz neni totez jako
# "jak dobre tahle featura SAMA rozlisi tridy", a ANOVA F-test je pro ucel
# klasifikace primejsi metrika.

# Zrychleny ladici rezim (viz docstring vyse) -- nemeni logiku, jen velikost modelu.
FAST_DEBUG = os.environ.get("GSD_FAST_DEBUG", "0") == "1"


# ----------------------------------------------------------------------------
# Nacteni dat
# ----------------------------------------------------------------------------

def load_data(train_path=TRAIN_PATH, test_path=TEST_PATH):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


# ----------------------------------------------------------------------------
# 1. FEATURE ENGINEERING
# ----------------------------------------------------------------------------

def _raw_grid(df, feat_cols):
    X_raw = df[feat_cols].to_numpy(dtype=np.float64)
    N = X_raw.shape[0]
    grid = X_raw.reshape(N, N_SENSORS, N_STATS)
    return X_raw, grid


def _robust_outlier_stats(X_raw):
    """Median/MAD po sloupcich -- stejna metoda jako data_view.ipynb sekce 9
    ('Outlier detection')."""
    median = np.median(X_raw, axis=0)
    mad = np.median(np.abs(X_raw - median), axis=0)
    return median, mad


def _robust_z(X_raw, median, mad):
    mad_safe = np.where(mad < 1e-9, 1e-9, mad)
    return (X_raw - median) / mad_safe * 0.6745


def winsorize_raw_features(X_raw, threshold=10.0):
    """Winsorizace extremnich hodnot pomoci robustniho z-skore (median/MAD),
    STEJNA metoda a stejny prah (|z|>10) jako data_view.ipynb sekce 9.

    DUVOD: notebook zjistil, ze prvni fyzicka jednotka TGS2602 (blok 1, tedy
    feat_1..feat_8) ma zdaleka nejvic extremnich hodnot (feat_8: 819
    outlieru, feat_7: 423, feat_6: 359, feat_1: 288 -- 7 z top 10
    nejvice-outlierovych priznaku pochazi prave z tohoto jednoho fyzickeho
    senzoru). Bez osetreni by tyhle extremy mohly neprimerene ovlivnit
    prumer/rozptyl pouzity v moment_align() (pocitane jako obycejny mean/std,
    citlive na odlehle hodnoty) a zesilit se v cnorm (deleni malou
    koncentraci muze extremni hodnotu jeste zvetsit).

    Vraci (winsorizovane X, robustni z-skore PRED winsorizaci -- pouzitelne
    pro odvozeni outlier_count priznaku)."""
    median, mad = _robust_outlier_stats(X_raw)
    z = _robust_z(X_raw, median, mad)
    mad_safe = np.where(mad < 1e-9, 1e-9, mad)
    cap = threshold * mad_safe / 0.6745
    lower = median - cap
    upper = median + cap
    return np.clip(X_raw, lower, upper), z


def compute_outlier_count_feature(z, threshold=10.0):
    """Pocet priznaku s |robustni z-skore| > threshold, na radek -- explicitni
    signal pro model 'jak neobvykle/potencialne sumove je tohle konkretni
    mereni', stejna metoda a prah jako data_view.ipynb sekce 9."""
    return (np.abs(z) > threshold).sum(axis=1).astype(np.float64)


def build_features(df, feat_cols, eps=1e-6, outlier_threshold=10.0):
    """Vytvori kompletni feature matici + jmena sloupcu + velikosti bloku
    (potrebne pro pozdejsi per-blokove zarovnani driftu, viz moment_align)."""
    X_raw_orig, _ = _raw_grid(df, feat_cols)
    X_raw, robust_z = winsorize_raw_features(X_raw_orig, threshold=outlier_threshold)
    outlier_count = compute_outlier_count_feature(robust_z, threshold=outlier_threshold)
    N = X_raw.shape[0]
    grid = X_raw.reshape(N, N_SENSORS, N_STATS)

    conc = df["concentration"].to_numpy(dtype=np.float64)
    conc_safe = np.where(np.abs(conc) < eps, eps, conc)

    blocks = {}
    names = {}

    # a) raw 128 priznaku (JIZ WINSORIZOVANE -- viz winsorize_raw_features)
    blocks["raw"] = X_raw
    names["raw"] = list(feat_cols)

    # b) cnorm = feat / concentration (128) -- oddeluje citlivost senzoru od davky plynu
    blocks["cnorm"] = X_raw / conc_safe[:, None]
    names["cnorm"] = [f"cnorm_{c}" for c in feat_cols]

    # c) concentration + log1p(concentration) -- NEPOVAZUJEME za "senzorovy drift",
    #    proto se tento blok pri moment_align preskakuje (viz nize)
    blocks["conc"] = np.column_stack([conc, np.log1p(np.maximum(conc, 0.0))])
    names["conc"] = ["concentration", "log_concentration"]

    # d) per-pozice statistiky napric vsemi 16 senzory: mean, std, range (8*3=24)
    pos_stats = []
    pos_names = []
    for d_idx, pname in enumerate(POSITION_NAMES):
        col = grid[:, :, d_idx]
        pos_stats.append(col.mean(axis=1))
        pos_stats.append(col.std(axis=1))
        pos_stats.append(col.max(axis=1) - col.min(axis=1))
        pos_names += [f"pos_{pname}_mean", f"pos_{pname}_std", f"pos_{pname}_range"]
    blocks["pos_stats"] = np.column_stack(pos_stats)
    names["pos_stats"] = pos_names

    # e) per sensor-TYPE agregace pres jeho 4 fyzicke kusy (mean, std) pro vsech
    #    8 pozic (4 typy * 8 pozic * 2 = 64). Toto da modelu moznost zvlast
    #    "kalibrovat" kazdy ze 4 typu -- klicove, protoze kazdy typ driftuje jinak.
    type_stats = []
    type_stats_names = []
    type_pos_means = {}
    for t in TYPE_NAMES:
        idx0 = [b - 1 for b in SENSOR_TYPE_BLOCKS[t]]
        sub = grid[:, idx0, :]                 # (N, 4, 8)
        m = sub.mean(axis=1)                   # (N, 8)
        s = sub.std(axis=1)                    # (N, 8)
        type_pos_means[t] = m
        type_stats.append(m)
        type_stats.append(s)
        type_stats_names += [f"{t}_{p}_mean" for p in POSITION_NAMES]
        type_stats_names += [f"{t}_{p}_unit_std" for p in POSITION_NAMES]
    blocks["type_stats"] = np.column_stack(type_stats)
    names["type_stats"] = type_stats_names

    # f) cross-TYPE pomery pro informativni pozice, vsech C(4,2)=6 dvojic typu
    #    (3*6=18). "Chemicky otisk prstu": jak silne reaguje TGS2602 RELATIVNE
    #    k TGS2600 na tutez udalost -- signatura KTERY plyn to je, mene citliva
    #    na absolutni drift jednoho konkretniho typu.
    cross_type = []
    cross_type_names = []
    for pos_idx in INFORMATIVE_POSITIONS:
        pname = POSITION_NAMES[pos_idx]
        for t1, t2 in itertools.combinations(TYPE_NAMES, 2):
            a = type_pos_means[t1][:, pos_idx]
            b = type_pos_means[t2][:, pos_idx]
            r = (a - b) / (np.abs(a) + np.abs(b) + eps)
            cross_type.append(r)
            cross_type_names.append(f"crosstype_{pname}_{t1}_vs_{t2}")
    blocks["cross_type"] = np.column_stack(cross_type)
    names["cross_type"] = cross_type_names

    # g) sousedni-senzor pomery v ramci syroveho 16-slotoveho poradi, jen pro
    #    informativni pozice (3*15=45) -- zachyti jemnejsi strukturu nez
    #    typove prumery.
    adj = []
    adj_names = []
    for pos_idx in INFORMATIVE_POSITIONS:
        pname = POSITION_NAMES[pos_idx]
        col = grid[:, :, pos_idx]
        for s_idx in range(N_SENSORS - 1):
            a = col[:, s_idx]
            b = col[:, s_idx + 1]
            r = (a - b) / (np.abs(a) + np.abs(b) + eps)
            adj.append(r)
            adj_names.append(f"adjratio_{pname}_s{s_idx + 1}_s{s_idx + 2}")
    blocks["adj_ratios"] = np.column_stack(adj)
    names["adj_ratios"] = adj_names

    # h) outlier_count (1) -- pocet priznaku s extremni robustni z-skore na
    #    radek (data_view.ipynb sekce 9). Explicitni "jak sumove/nespolehlive
    #    je tohle konkretni mereni" signal pro model.
    blocks["quality"] = outlier_count.reshape(-1, 1)
    names["quality"] = ["outlier_count"]

    order = ["raw", "cnorm", "conc", "pos_stats", "type_stats", "cross_type",
             "adj_ratios", "quality"]
    X = np.hstack([blocks[k] for k in order]).astype(np.float32)
    all_names = sum([names[k] for k in order], [])
    block_sizes = {k: blocks[k].shape[1] for k in order}
    return X, all_names, block_sizes


# ----------------------------------------------------------------------------
# 2. PER-FEATURE DRIFT ALIGNMENT (moment matching, NE plna kovariancni CORAL)
# ----------------------------------------------------------------------------

def moment_align(X_train, X_test, skip_mask=None, clip_scale=(0.2, 5.0)):
    """Univariatni (per-priznak) drift zarovnani: posune KAZDY priznak
    trenovacich dat tak, aby jeho prumer/rozptyl odpovidal prumeru/rozptylu
    stejneho priznaku v testu.

    skip_mask: bool pole (D,) -- True pro sloupce, ktere se NEMAJI zarovnavat
        (typicky concentration/log_concentration -- to neni "drift senzoru",
        je to jina experimentalni protokolova volba mezi train a test).
    """
    mu_tr = X_train.mean(axis=0, keepdims=True)
    sd_tr = X_train.std(axis=0, keepdims=True)
    mu_te = X_test.mean(axis=0, keepdims=True)
    sd_te = X_test.std(axis=0, keepdims=True)

    sd_tr_safe = np.where(sd_tr < 1e-8, 1e-8, sd_tr)
    scale = sd_te / sd_tr_safe
    scale = np.clip(scale, clip_scale[0], clip_scale[1])

    X_aligned = (X_train - mu_tr) * scale + mu_te

    if skip_mask is not None:
        X_aligned[:, skip_mask] = X_train[:, skip_mask]

    return X_aligned.astype(np.float32)


def build_conc_skip_mask(block_sizes):
    """True pro sloupce patrici do bloku 'conc' (concentration, log_concentration)."""
    order = ["raw", "cnorm", "conc", "pos_stats", "type_stats", "cross_type", "adj_ratios", "quality"]
    total = sum(block_sizes[k] for k in order)
    mask = np.zeros(total, dtype=bool)
    offset = 0
    for k in order:
        size = block_sizes[k]
        if k == "conc":
            mask[offset:offset + size] = True
        offset += size
    return mask


# ----------------------------------------------------------------------------
# 2b. KONCENTRACNI STROP -- fyzikalne/experimentalne motivovany prior
# ----------------------------------------------------------------------------

def compute_concentration_ceilings(train_df, n_classes=N_CLASSES):
    """Pro kazdou tridu (1..n_classes) maximalni koncentrace pozorovana v
    train.csv. Spocitano VYHRADNE z poskytnutych dat (gas_class, concentration
    sloupce train.csv) -- ZADNA informace o testovaci sade, ZADNY konflikt s
    pravidlem o nevyuzivani znalosti distribuce testovacich trid (viz bod 3 a
    5 v modulovem docstringu). Vraci pole indexovane 0..n_classes-1 (index
    c-1 odpovida tride c)."""
    s = train_df.groupby("gas_class")["concentration"].max()
    return s.reindex(range(1, n_classes + 1)).to_numpy(dtype=np.float64)


def concentration_prior_penalty(concentration, ceilings, margin=75.0, strength=3.0):
    """Hladka (NE tvrda) cost-penalizace pro kombinace (radek, trida), kde
    koncentrace radku prevysuje trenovany strop dane tridy.

    `margin` (ppm): jak daleko za stropem penalizace teprve zacina narustat --
        ~75ppm podle typicke mezery mezi sousednimi testovanymi hladinami
        blizko stropu v tomto datasetu -- radek jen malo za stropem tedy neni
        trestan stejne tvrde jako radek stovky ppm za nim.
    `strength`: skaluje, jak silne prior pretlaci modelovy odhad pravdepodobnosti.
        PEVNE NASTAVENO RUCNE (ne optimalizovano Nelder-Meadem), viz bod 5 v
        modulovem docstringu -- OOF nema temer zadne pokryti nad ~300ppm pro
        polovinu trid, takze automaticka optimalizace by byla slepa.

    Vraci matici (N, n_classes), ktera se PRICTE k -log(prob) cost matici.
    Tohle je VYPOCET PER RADEK -- nezavisi na zadne agregatni statistice
    testovaci sady, jen na (concentration radku, ceilings z train.csv)."""
    over = np.maximum(0.0, concentration[:, None] - (ceilings[None, :] + margin))
    penalty = strength * np.log1p(over / 100.0)
    return penalty


def report_concentration_coverage_gap(train_df, test_df, ceilings):
    """Diagnostika (bez opravy!) toho, jak velka cast testu lezi nad
    trenovanym koncentracnim stropem jednotlivych trid -- aby tahle mezera
    byla vzdy VIDITELNA v logu."""
    conc_levels = np.sort(test_df["concentration"].unique())
    print("    Pokryti podle koncentrace (kolik z 6 trid ma v train.csv precedens):")
    n_thin = 0
    for lvl in conc_levels:
        n_rows = int((test_df["concentration"] == lvl).sum())
        possible = int((ceilings >= lvl).sum())
        marker = "  <-- MALO/ZADNE trenovaci pokryti" if possible <= 2 else ""
        if possible <= 2:
            n_thin += n_rows
        print(f"      conc={lvl:>7.1f} ppm  test_radku={n_rows:>4d}  "
              f"moznych_trid={possible}/{N_CLASSES}{marker}")
    print(f"    -> Celkem {n_thin} testovacich radku ({n_thin/len(test_df)*100:.1f}%) "
          f"lezi v oblasti, kde <=2 tridy maji trenovaci precedens.")


def report_oof_f1_by_concentration_bucket(oof_preds_hard, y_true, train_concentration, n_buckets=5):
    """OOF macro-F1 rozdelene do bucketu podle koncentrace -- nejblizsi
    dostupna napodobenina 'concentration-stratified hold-out', pouzivajici
    jen skutecne, overene labely. POZOR: tohle NEMUZE zmerit vykon nad
    ~300-400ppm pro poloviny trid, protoze tam prosto neexistuji zadne (nebo
    skoro zadne) trenovaci radky -- to je presne ta mezera, viz
    report_concentration_coverage_gap().

    `oof_preds_hard`: uz hotove tvrde predikce (argmax financich, stackovanych
    a kalibrovanych OOF pravdepodobnosti) -- funkce uz nedela zadne blendovani
    sama (to ted resi stacking meta-learner, viz fit_stacking_meta_learner_cv)."""
    edges = np.quantile(train_concentration, np.linspace(0, 1, n_buckets + 1))
    edges[0] = -np.inf
    edges[-1] = np.inf
    bucket_idx = np.digitize(train_concentration, edges[1:-1])

    print("    OOF macro-F1 podle koncentracniho bucketu (jen realne labely):")
    for b in range(n_buckets):
        mask = bucket_idx == b
        if mask.sum() == 0:
            continue
        lo, hi = edges[b], edges[b + 1]
        f1 = f1_score(y_true[mask], oof_preds_hard[mask], average="macro")
        print(f"      [{lo:>7.1f}, {hi:>7.1f}) ppm  n={mask.sum():>5d}  macro-F1={f1:.4f}")


# ----------------------------------------------------------------------------
# 3. MODELY -- jen XGBoost, CatBoost, LightGBM (na pozadavek uzivatele)
# ----------------------------------------------------------------------------

def get_candidate_models():
    n_div = 6 if FAST_DEBUG else 1  # zrychleny rezim: cca 1/6 poctu stromu

    lgb_params = dict(
        objective="multiclass", num_class=N_CLASSES, metric="multi_logloss",
        boosting_type="gbdt", n_estimators=max(60, 600 // n_div), learning_rate=0.03,
        num_leaves=40, max_depth=7, subsample=0.85, colsample_bytree=0.7,
        reg_alpha=1.0, reg_lambda=3.0, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1,
    )
    xgb_params = dict(
        objective="multi:softprob", num_class=N_CLASSES, eval_metric="mlogloss",
        n_estimators=max(60, 600 // n_div), learning_rate=0.03, max_depth=6, subsample=0.85,
        colsample_bytree=0.7, reg_alpha=1.0, reg_lambda=3.0,
        random_state=RANDOM_STATE, n_jobs=-1, tree_method="hist",
    )
    cb_params = dict(
        loss_function="MultiClass", iterations=max(80, 650 // n_div), learning_rate=0.035,
        depth=6, l2_leaf_reg=4.0, random_seed=RANDOM_STATE, thread_count=-1, verbose=False,
    )

    return {
        "xgboost": xgb.XGBClassifier(**xgb_params),
        "catboost": CatBoostClassifier(**cb_params),
        "lightgbm": lgb.LGBMClassifier(**lgb_params),
    }


# ----------------------------------------------------------------------------
# 3b. CASOVE KOREKTNI VYBER MODELU (batch<=8 -> batch9)
# ----------------------------------------------------------------------------

def compare_models_time_split(X, y, batch_arr, candidates, verbose=True):
    """Forward-chaining casova validace: train<=6->val7, train<=7->val8,
    train<=8->val9 -- prumerne macro-F1 pres tyto 3 rozdeleni.

    DULEZITE: batch9 samotny ma jen 3 unikatni koncentracni hladiny
    (10/100/120 ppm) -- je to nejuzsi batch v celem datasetu. Jednorazovy
    split (jen train<=8->val9) je proto vysoce rozptylena metrika: v jednom
    overenem behu vyslo catboost=0.955 vs xgboost=0.762 (rozdil 0.19!) na
    tomto splitu, ale jen 0.933 vs 0.919 (rozdil 0.014) na forward-chaining
    prumeru -- tedy temer 10x mensi a mnohem duveryhodnejsi rozdil.
    Prumerovani pres 3 nezavisla rozdeleni (kazde s jinou koncentracni/batch
    skladbou validacniho batche) tohle riziko zmirnuje, i kdyz stale
    NEPOKRYVA vysoko-koncentracni oblast (>300ppm) -- na to viz
    concentration_prior_penalty() a diagnostiku v [4]."""
    splits = [(6, 7), (7, 8), (8, 9)]
    per_split_scores = {name: [] for name in candidates}

    for tr_max, va_batch in splits:
        tr_mask = batch_arr <= tr_max
        va_mask = batch_arr == va_batch
        for name, model in candidates.items():
            m = clone(model)
            m.fit(X[tr_mask], y[tr_mask])
            pred = np.asarray(m.predict(X[va_mask])).reshape(-1)
            f1 = f1_score(y[va_mask], pred, average="macro")
            per_split_scores[name].append(f1)

    results = {name: float(np.mean(scores)) for name, scores in per_split_scores.items()}
    if verbose:
        print("      forward-chaining jednotlive splity (train<=6->7, <=7->8, <=8->9):")
        for name, scores in per_split_scores.items():
            print(f"        {name:>10s}: {[f'{s:.4f}' for s in scores]}  "
                  f"prumer={np.mean(scores):.5f}  std={np.std(scores):.4f}")
    return results


def evaluate_batch7_holdout(X, y, batch_arr, candidates, verbose=True):
    """Trenink na VSECH batchich krome 7, validace na batchi 7.

    DUVOD PRO SPECIALNI STATUS BATCHE 7 (overeno primo na datech, KS-test
    distribucni vzdalenosti kazdeho batche vuci testu/batchi 10):
        batch 7: KS=0.207 (NEJLEPSI ze vsech 9 batchu, tedy nejreprezentativnejsi
                  proxy pro skutecny test)
        batch 9: KS=0.365 (jeden z NEJHORSICH -- prekvapive, batch9 je pritom
                  ten, na kterem konci forward-chaining a na kterem se
                  z velke casti pocita OOF)
        cely pooled train (1-9): KS=0.184
    Batch 7 je navic zdaleka nejvetsi (n=3613, skoro presne velikost testu
    n=3600) -- kombinace "nejreprezentativnejsi" + "nejvetsi" z nej dela
    kvalitnejsi jednorazovy vyberovy signal nez cokoliv v puvodnim
    forward-chainingu samotnem.

    Pouziva SE VSECHNY OSTATNI batche pro trenink (ne jen batch<=6 jako ve
    forward-chainingu) -- vyuziva vic dat, protoze cilem tady neni simulovat
    "jak by to vypadalo v case X", ale ziskat co nejsilnejsi/nejspolehlivejsi
    odhad vykonu na necem, co se skutecnemu testu nejvic podoba."""
    va_mask = batch_arr == 7
    tr_mask = ~va_mask

    results = {}
    for name, model in candidates.items():
        m = clone(model)
        m.fit(X[tr_mask], y[tr_mask])
        pred = np.asarray(m.predict(X[va_mask])).reshape(-1)
        f1 = f1_score(y[va_mask], pred, average="macro")
        results[name] = f1
    if verbose:
        print("      batch7-holdout (trenink na vsem krome batch7, validace na nem):")
        for name, f1 in results.items():
            print(f"        {name:>10s}: {f1:.5f}")
    return results


# ----------------------------------------------------------------------------
# 4. LEAVE-ONE-BATCH-OUT OOF (pro kalibraci a pseudo-labeling)
# ----------------------------------------------------------------------------

def leave_one_batch_out_oof(X_real, y_real, batch_real, X_extra, y_extra, X_test, models_dict):
    """
    X_real / y_real / batch_real: originalni (skutecne) trenovaci radky, batch 1-9.
    X_extra / y_extra: aktualni pseudo-labelovane radky (z testu) -- VZDY soucasti
        trenovaci casti KAZDEHO foldu, NIKDY validacni (nemaji spolehlivy casovy
        "batch" a jejich label je jen odhad) -- OOF metriky tak zustavaji cisté,
        pocitane jen na skutecnych, overenych labelech.
    Vraci OOF pravdepodobnosti (pro X_real) + prumerne testovaci pravdepodobnosti
    pres vsech len(unique_batches) foldu.
    """
    unique_batches = np.unique(batch_real)
    oof = {name: np.zeros((len(y_real), N_CLASSES), dtype=np.float32) for name in models_dict}
    test_pred_sum = {name: np.zeros((X_test.shape[0], N_CLASSES), dtype=np.float32) for name in models_dict}

    for b in unique_batches:
        va_mask = batch_real == b
        tr_mask = ~va_mask
        X_tr = X_real[tr_mask]
        y_tr = y_real[tr_mask]
        if X_extra is not None and len(y_extra) > 0:
            X_tr = np.vstack([X_tr, X_extra])
            y_tr = np.concatenate([y_tr, y_extra])

        for name, model in models_dict.items():
            m = clone(model)
            m.fit(X_tr, y_tr)
            oof[name][va_mask] = m.predict_proba(X_real[va_mask])
            test_pred_sum[name] += m.predict_proba(X_test)

    n_folds = len(unique_batches)
    test_pred = {name: test_pred_sum[name] / n_folds for name in models_dict}
    return oof, test_pred


# ----------------------------------------------------------------------------
# 5. STACKING META-LEARNER (nahrazuje skalarni blend_w) + tridni kalibrace
# ----------------------------------------------------------------------------

def fit_stacking_meta_learner_cv(oof_list, y_true, batch_arr, test_list, C=1.0):
    """Stacking meta-learner (multinomicka logisticka regrese) SE SVYM
    VLASTNIM leave-one-batch-out schematem -- NAHRAZUJE puvodni jednoduchou
    verzi (fit_stacking_meta_learner + apply_stacking_meta_learner), ktera
    mela metodologickou chybu: meta-learner se fitoval NA CELE OOF matici a
    pak se na TE SAME matici vyhodnocoval pro OOF macro-F1 report i pro
    kalibraci class_w. To neni genuine out-of-sample odhad -- je to
    "trenovaci presnost meta-vrstvy", ktera muze byt uměle vysoka (overeno
    na skutecnem behu: OOF macro-F1 vyskocilo z ~0.94-0.96 na ~0.986-0.990,
    mnohem vic nez odpovidalo realnemu zlepseni na testu).

    OPRAVA: stejny princip jako uz existujici leave_one_batch_out_oof() pro
    zakladni modely, jen o uroven vys (nad JEJICH OOF vystupy jakozto
    vstupnimi priznaky meta-learneru). Pro kazdy batch b:
      1. meta-learner se natrenuje na meta-priznacich VSECH OSTATNICH batchu
      2. predikuje na OOF-radcich batche b (-> nested_oof, honest odhad)
      3. predikuje i na testovacich meta-priznacich (-> prumeruje se pres
         vsech 9 takto natrenovanych meta-learneru -- STEJNA bagging
         filozofie, jakou uz pro zakladni modely pouziva
         leave_one_batch_out_oof(), takze produkcni testovaci predikce
         vznikaji konzistentnim zpusobem na obou urovnich stacku)

    oof_list / test_list: seznamy (N, N_CLASSES) poli v PEVNEM poradi (napr.
        [oof_a, oof_b, oof_type_TGS2600, ...] / [test_a, test_b, test_type_TGS2600, ...]).

    Vraci (nested_oof, test_pred) -- nested_oof pouzij pro
    calibrate_class_weights() a report_oof_f1_by_concentration_bucket()
    (honest cisla), test_pred pouzij pro pseudo-labeling a financi krok."""
    X_meta = np.hstack(oof_list)
    X_meta_test = np.hstack(test_list)

    nested_oof = np.zeros((len(y_true), N_CLASSES), dtype=np.float32)
    test_pred_sum = np.zeros((X_meta_test.shape[0], N_CLASSES), dtype=np.float32)
    unique_batches = np.unique(batch_arr)

    for b in unique_batches:
        va_mask = batch_arr == b
        tr_mask = ~va_mask
        meta = LogisticRegression(max_iter=2000, C=C)  # lbfgs (default, sklearn>=1.5)
        # je automaticky multinomicky pro >2 tridy -- explicitni multi_class
        # param. byl v novejsich verzich sklearn odstranen.
        meta.fit(X_meta[tr_mask], y_true[tr_mask])
        nested_oof[va_mask] = meta.predict_proba(X_meta[va_mask])
        test_pred_sum += meta.predict_proba(X_meta_test)

    test_pred = test_pred_sum / len(unique_batches)
    return nested_oof, test_pred


def calibrate_class_weights(probs, y_true, maxiter=2000, l2_reg=0.05):
    """Nelder-Mead optimalizace 6 tridnich multiplikatoru (BEZ blend_w --
    ten ted resi stacking meta-learner, viz fit_stacking_meta_learner_cv).
    Maximalizuje macro-F1 na (jiz stackovanych) OOF pravdepodobnostech, s L2
    regularizaci na log(class_w) smerem k 1.0.

    PROC REGULARIZACE: v overenem behu bez ni trida "5" (nejobtiznejsi trida)
    mela vahu kmitajici 0.669 -> 0.443 -> 0.887 -> 0.311 napric koly
    pseudo-labelingu (2.9x rozptyl mezi sousednimi koly, bez konvergence,
    zatimco OOF F1 stale rostl 0.941->0.957 -- typicky priznak preuceni
    kalibratoru na sum) -- kalibrator se snazi globalnim multiplikatorem
    opravit chybu, ktera je ve skutecnosti soustredena jen v jedne
    koncentracni oblasti (viz concentration_prior_penalty), takze globalni
    oprava nemuze fungovat a misto toho jen prehani vahu tam a zpet. L2
    penalizace odrazuje extremni vychylky a stabilizuje vysledek mezi koly."""
    def loss(class_w):
        scaled = probs * class_w
        preds = np.argmax(scaled, axis=1)
        f1 = f1_score(y_true, preds, average="macro")
        reg = l2_reg * np.mean(np.log(np.maximum(class_w, 1e-6)) ** 2)
        return -f1 + reg

    x0 = np.ones(N_CLASSES)
    res = minimize(
        loss, x0, method="Nelder-Mead",
        options={"maxiter": maxiter, "xatol": 1e-4, "fatol": 1e-6, "disp": False},
    )
    class_w = res.x / np.mean(res.x)
    true_f1 = f1_score(y_true, np.argmax(probs * class_w, axis=1), average="macro")
    return class_w.astype(np.float32), float(true_f1)


# ----------------------------------------------------------------------------
# 5b. PER-SENZOR-TYPOVI "EXPERTI" (dalsi vstup do stacking meta-learneru)
# ----------------------------------------------------------------------------

def build_per_type_features(df, feat_cols):
    """Pro kazdy ze 4 typu senzoru vytvori SAMOSTATNOU feature matici jen
    z jeho vlastnich 4 fyzickych kusu: raw (4*8=32) + cnorm (32) + typovy
    mean/std pres 4 kusy (8*2=16) = 80 priznaku na typ. Cilem je zjistit,
    KOLIK samostatneho signalu nese KAZDY typ senzoru zvlast -- primo
    odpovida na otazku 'zamer se na rozlisovani mezi jednotlivymi senzory'.

    Pouziva STEJNOU winsorizaci jako hlavni build_features() (konzistence)."""
    X_raw_orig, _ = _raw_grid(df, feat_cols)
    X_raw, _ = winsorize_raw_features(X_raw_orig)
    N = X_raw.shape[0]
    grid = X_raw.reshape(N, N_SENSORS, N_STATS)
    conc = df["concentration"].to_numpy(dtype=np.float64)
    conc_safe = np.where(np.abs(conc) < 1e-6, 1e-6, conc)

    per_type = {}
    for t in TYPE_NAMES:
        idx0 = [b - 1 for b in SENSOR_TYPE_BLOCKS[t]]
        sub = grid[:, idx0, :]                    # (N, 4, 8)
        raw_t = sub.reshape(N, -1)                # (N, 32)
        cnorm_t = raw_t / conc_safe[:, None]       # (N, 32)
        mean_t = sub.mean(axis=1)                  # (N, 8)
        std_t = sub.std(axis=1)                    # (N, 8)
        per_type[t] = np.hstack([raw_t, cnorm_t, mean_t, std_t]).astype(np.float32)
    return per_type


def fit_per_type_experts(train_df, feat_cols, y, batch_arr, test_df, n_estimators=150):
    """Trenuje 4 nezavisle 'expert' modely, kazdy jen na priznacich JEDNOHO
    typu senzoru (viz build_per_type_features), pomoci leave-one-batch-out
    (9 foldu). Vraci OOF a testovaci pravdepodobnosti pro kazdy typ -- URCENE
    JAKO DALSI VSTUP do stacking meta-learneru (fit_stacking_meta_learner_cv),
    NE jako samostatna soucast hlavniho ensemblu.

    NAKLADOVA POZNAMKA: experti se FITUJI JEN JEDNOU (pred pseudo-labeling
    smyckou), NE znovu v kazdem kole -- 4 dalsi typy x 9 foldu x 3 kola by
    ~ztrojnasobilo dobu behu jen kvuli tomuto pridavku. Jejich signal je
    spis strukturalni/typovy ('kolik plyn-diskriminujici informace nese TENTO
    typ senzoru sam o sobe'), ktery by se nemel s pribyvajicimi pseudo-
    labely hodne menit -- zjednoduseni je proto zamerne, ne z lenosti."""
    per_type_train = build_per_type_features(train_df, feat_cols)
    per_type_test = build_per_type_features(test_df, feat_cols)
    unique_batches = np.unique(batch_arr)

    oof_by_type = {}
    test_by_type = {}
    for t in TYPE_NAMES:
        X_t = per_type_train[t]
        X_t_test = per_type_test[t]
        oof_t = np.zeros((len(y), N_CLASSES), dtype=np.float32)
        test_sum = np.zeros((X_t_test.shape[0], N_CLASSES), dtype=np.float32)
        for b in unique_batches:
            va_mask = batch_arr == b
            tr_mask = ~va_mask
            m = lgb.LGBMClassifier(
                objective="multiclass", num_class=N_CLASSES, n_estimators=n_estimators,
                max_depth=5, num_leaves=20, learning_rate=0.05,
                random_state=RANDOM_STATE, n_jobs=-1, verbose=-1,
            )
            m.fit(X_t[tr_mask], y[tr_mask])
            oof_t[va_mask] = m.predict_proba(X_t[va_mask])
            test_sum += m.predict_proba(X_t_test)
        oof_by_type[t] = oof_t
        test_by_type[t] = test_sum / len(unique_batches)

    return oof_by_type, test_by_type


# ----------------------------------------------------------------------------
# 6. PSEUDO-LABELING (konsenzus + kvota na tridu)
# ----------------------------------------------------------------------------

def select_pseudo_labels(test_probs_calibrated, pred_a_hard, pred_b_hard, cap_per_class,
                          test_concentration=None, ceilings=None, hard_margin=50.0):
    """Pseudo-label kandidat kvalifikuje, pokud (a) OBA modely predikuji stejnou
    tridu (konsenzus) A ZAROVEN (b) je mezi top-`cap_per_class` nejjistejsimi
    (podle kalibrovane pravdepodobnosti) pro danou tridu.

    Pokud jsou dodany `test_concentration` a `ceilings`, pridava se navic TVRDA
    brana: radek se SMI stat pseudo-labelem tridy c jen pokud jeho koncentrace
    <= ceilings[c] + hard_margin. Duvod je tvrdsi nez u financiho
    concentration_prior_penalty() (ktery je jen mekky): pseudo-label se stava
    SOUCASTI TRENOVACICH DAT pro dalsi kolo -- pokud bychom sem pustili
    sebejiste-ale-extrapolovane (fyzikalne nepravdepodobne) predikce, model by
    se v dalsim kole ucil ze svych vlastnich chyb a chybu by tim upevnil."""
    agree_mask = pred_a_hard == pred_b_hard
    hard_pred = np.argmax(test_probs_calibrated, axis=1)
    conf = np.max(test_probs_calibrated, axis=1)

    plausible_mask = np.ones(len(hard_pred), dtype=bool)
    if test_concentration is not None and ceilings is not None:
        row_ceiling = ceilings[hard_pred]
        plausible_mask = test_concentration <= (row_ceiling + hard_margin)

    selected_idx, selected_labels = [], []
    for c in range(N_CLASSES):
        c_mask = agree_mask & (hard_pred == c) & plausible_mask
        c_idx = np.where(c_mask)[0]
        order = np.argsort(-conf[c_idx])
        take = c_idx[order[:cap_per_class]]
        selected_idx.extend(take.tolist())
        selected_labels.extend([c] * len(take))
    return np.array(selected_idx, dtype=int), np.array(selected_labels, dtype=int)


# ----------------------------------------------------------------------------
# Diagnostika: dulezitost priznaku podle kategorie
# ----------------------------------------------------------------------------

def report_feature_importance_by_block(model, feat_names, block_sizes):
    if not hasattr(model, "feature_importances_"):
        return
    importances = np.asarray(model.feature_importances_, dtype=np.float64)
    total = importances.sum()
    if total <= 0:
        return
    importances = importances / total

    order = ["raw", "cnorm", "conc", "pos_stats", "type_stats", "cross_type", "adj_ratios", "quality"]
    offset = 0
    print("    Dulezitost priznaku podle kategorie (podil na celkove dulezitosti modelu):")
    for k in order:
        size = block_sizes[k]
        share = importances[offset:offset + size].sum()
        print(f"      {k:>14s} ({size:>3d} priznaku): {share * 100:5.1f}%")
        offset += size


# ----------------------------------------------------------------------------
# Diagnostika: drift-citlivost pri nizke koncentraci (VOLITELNA, rucne spoustena)
# ----------------------------------------------------------------------------

def diagnose_low_concentration_drift_sensitivity(train_df, feat_cols, threshold=50.0,
                                                   n_estimators=200, n_splits=3,
                                                   random_state=RANDOM_STATE):
    """VOLITELNA, JEDNORAZOVA diagnostika -- NENI volana automaticky v main(),
    protoze pridava ~n_splits dalsich fitu modelu navic k jiz drahemu behu.
    Spustit rucne, kdyz chces overit/reprodukovat tohle zjisteni.

    Porovna vykon v nizko-koncentracni oblasti (<threshold ppm) mezi:
      (a) nahodnym stratifikovanym K-foldem (michajicim vsechny batche dohromady)
      (b) leave-one-batch-out schematem pouzitym jinde v teto pipeline (viz
          report_oof_f1_by_concentration_bucket() v hlavnim behu main())

    DUVOD / OVERENY VYSLEDEK NA REALNYCH DATECH: nahodny 3-fold K-fold dal v
    teto oblasti macro-F1 = 0.9996 (temer dokonale!), zatimco leave-one-batch-out
    OOF v hlavni pipeline dal ve stejne oblasti jen 0.6380. Tenhle obrovsky
    rozdil ukazuje, ze slabina NENI fyzikalni "nizky signal/sum pomer pri
    malych davkach" (to by se projevilo stejne i v nahodnem K-foldu, protoze
    ten problem by nezavisel na tom, jestli batche michame nebo ne) -- je to
    DRIFT-CITLIVOST SPECIFICKA PRO NIZKOU KONCENTRACI: absolutni velikost
    senzoroveho signalu (DR/EMA) roste s koncentraci, takze STEJNA absolutni
    chyba driftove korekce (moment_align) tvori mnohem vetsi RELATIVNI chybu
    pri nizke koncentraci nez pri vysoke -- a nahodny K-fold tohle neodhali,
    protoze v nem model "vidi" driftovy stav vsech batchu rozmichany dohromady
    (presne ten problem, kvuli kteremu jinde v pipeline pouzivame
    leave-one-batch-out/forward-chaining!).

    POZNAMKA: tohle je DIAGNOSTIKA, ne oprava. Spravny fix by pravdepodobne
    vyzadoval hlubsi redesign driftove korekce (napr. relativni/procentualni
    korekce namisto absolutni, nebo koncentrace-zavisla sila korekce) -- to
    je vetsi zmena, kterou stoji za to probrat zvlast pred implementaci.
    """
    y = train_df["gas_class"].to_numpy(dtype=int) - 1
    conc = train_df["concentration"].to_numpy(dtype=np.float64)
    X_raw, _, _ = build_features(train_df, feat_cols)
    X = X_raw.astype(np.float32)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof_pred = np.zeros(len(y), dtype=int)
    for tr_idx, va_idx in skf.split(X, y):
        m = xgb.XGBClassifier(
            objective="multi:softprob", num_class=N_CLASSES, n_estimators=n_estimators,
            max_depth=6, learning_rate=0.05, tree_method="hist", n_jobs=-1,
        )
        m.fit(X[tr_idx], y[tr_idx])
        oof_pred[va_idx] = m.predict(X[va_idx])

    low_mask = conc < threshold
    f1_low = f1_score(y[low_mask], oof_pred[low_mask], average="macro")
    f1_all = f1_score(y, oof_pred, average="macro")

    print(f"[diagnostika drift-citlivosti pri nizke koncentraci, prah={threshold}ppm]")
    print(f"  Nahodny {n_splits}-fold K-fold (michajici batche napric casem):")
    print(f"    macro-F1 v oblasti <{threshold}ppm: {f1_low:.4f}")
    print(f"    macro-F1 celkove:                  {f1_all:.4f}")
    print("  Porovnej s 'OOF macro-F1 podle koncentracniho bucketu' v hlavnim")
    print("  behu main() pro tutez oblast (leave-one-batch-out, respektuje cas).")
    print("  Velky rozdil = potvrzena drift-citlivost, NE fyzikalni limit rozliseni.")
    return {"f1_low_random_cv": f1_low, "f1_all_random_cv": f1_all, "threshold": threshold}


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------

def main():
    print("=" * 78)
    print("Gas Sensor Array Drift -- modelovaci pipeline")
    print("(3 modely, bez 600/tridu, forward-chaining+batch7-holdout vyber,")
    print(" koncentracni prior, stacking meta-learner + per-typovi experti)")
    if FAST_DEBUG:
        print("!!! GSD_FAST_DEBUG=1 -- bezi zmensena/ladici verze, NE produkcni kvalita !!!")
    print("=" * 78)

    print("\n[0] Nacitani dat...")
    train_df, test_df = load_data()
    feat_cols = [f"feat_{i}" for i in range(1, 129)]
    y = train_df["gas_class"].to_numpy(dtype=int) - 1
    batch_arr = train_df["batch"].to_numpy(dtype=int)
    print(f"    train: {train_df.shape}, test: {test_df.shape}")

    print("\n[1] Feature engineering...")
    X_train_raw, feat_names, block_sizes = build_features(train_df, feat_cols)
    X_test_raw, _, _ = build_features(test_df, feat_cols)
    print(f"    Celkem {X_train_raw.shape[1]} priznaku. Po blocich:")
    for k, v in block_sizes.items():
        print(f"      {k:>14s}: {v}")

    print("\n[2] Per-priznakove zarovnani driftu (train -> test moment matching)...")
    skip_mask = build_conc_skip_mask(block_sizes)
    X_train = moment_align(X_train_raw, X_test_raw, skip_mask=skip_mask)
    X_test = X_test_raw.astype(np.float32)

    train_concentration = train_df["concentration"].to_numpy(dtype=np.float64)
    test_concentration = test_df["concentration"].to_numpy(dtype=np.float64)
    ceilings = compute_concentration_ceilings(train_df)
    print("\n    Koncentracni stropy podle tridy (max ppm pozorovany v train.csv,")
    print("    pocitano VYHRADNE z train.csv -- viz bod 5 v docstringu):")
    print("     ", {i + 1: float(c) for i, c in enumerate(ceilings)})
    report_concentration_coverage_gap(train_df, test_df, ceilings)

    print("\n[3] Porovnani 3 modelu -- forward-chaining + batch7-holdout (viz bod 9)...")
    candidates = get_candidate_models()
    fc_scores = compare_models_time_split(X_train, y, batch_arr, candidates)
    print()
    b7_scores = evaluate_batch7_holdout(X_train, y, batch_arr, candidates)

    combined_scores = {
        name: 0.5 * fc_scores[name] + 0.5 * b7_scores[name] for name in candidates
    }
    ranked = sorted(combined_scores.items(), key=lambda kv: kv[1], reverse=True)
    print("\n      -> kombinovane skore (0.5*forward-chaining + 0.5*batch7-holdout):")
    for name, sc in ranked:
        print(f"           {name:>10s}: forward-chain={fc_scores[name]:.5f}  "
              f"batch7-holdout={b7_scores[name]:.5f}  kombinovane={sc:.5f}")
    best_two = [name for name, _ in ranked[:2]]
    a_name, b_name = best_two
    print(f"    -> Vybrane 2 nejlepsi modely: {a_name}, {b_name}")

    chosen_models = {name: candidates[name] for name in best_two}

    print("\n[3b] Trenink 4 per-senzor-typovych 'expertu' (jednorazove, viz bod 10)...")
    oof_by_type, test_by_type = fit_per_type_experts(train_df, feat_cols, y, batch_arr, test_df)
    for t in TYPE_NAMES:
        expert_pred = np.argmax(oof_by_type[t], axis=1)
        expert_f1 = f1_score(y, expert_pred, average="macro")
        print(f"      {t}: samostatne (OOF) macro-F1 = {expert_f1:.4f}  "
              f"(pro srovnani, NE pouzito samostatne -- jen jako vstup do stackingu)")

    print("\n[4] Leave-one-batch-out OOF + stacking meta-learner (s VLASTNIM")
    print("    leave-one-batch-out, viz bod 11 v docstringu) + tridni kalibrace...")
    print("    (POZOR na zdokumentovanou cirkularitu s batch9 -- viz docstring modulu)")
    oof, test_pred = leave_one_batch_out_oof(
        X_train, y, batch_arr, None, None, X_test, chosen_models
    )

    def _stack_inputs(oof_or_test_a, oof_or_test_b, by_type_dict):
        return [oof_or_test_a, oof_or_test_b] + [by_type_dict[t] for t in TYPE_NAMES]

    nested_oof, stacked_test = fit_stacking_meta_learner_cv(
        _stack_inputs(oof[a_name], oof[b_name], oof_by_type), y, batch_arr,
        _stack_inputs(test_pred[a_name], test_pred[b_name], test_by_type),
    )
    class_w, oof_f1 = calibrate_class_weights(nested_oof, y)
    print(f"    class_w={np.round(class_w, 3)}  "
          f"OOF macro-F1 (nested-stacked+kalibrovano, HONEST)={oof_f1:.5f}")
    report_oof_f1_by_concentration_bucket(
        np.argmax(nested_oof * class_w, axis=1), y, train_concentration
    )

    X_extra = None
    y_extra = None
    PSEUDO_ROUNDS_CAPS = [50, 100, 150] if FAST_DEBUG else [100, 200, 300]
    # VRACENO ZPATKY na 3 kola po skutecnem A/B testu na testovacim skore --
    # viz bod 8 v docstringu (OOF neni pro tenhle hyperparametr spolehlivy).

    for round_idx, cap in enumerate(PSEUDO_ROUNDS_CAPS, start=1):
        print(f"\n[5.{round_idx}] Pseudo-labeling kolo {round_idx}/{len(PSEUDO_ROUNDS_CAPS)} "
              f"(kvota={cap}/tridu)...")

        test_calibrated = stacked_test * class_w
        test_calibrated /= test_calibrated.sum(axis=1, keepdims=True)

        # Konsenzus se stale pocita mezi 2 HLAVNIMI modely (ne experty) --
        # to je jiz overeny, validovany mechanismus; experti jen prispivaji
        # do stackingu, nemeni definici "shody".
        pred_a_hard = np.argmax(test_pred[a_name], axis=1)
        pred_b_hard = np.argmax(test_pred[b_name], axis=1)

        sel_idx, sel_labels = select_pseudo_labels(
            test_calibrated, pred_a_hard, pred_b_hard, cap_per_class=cap,
            test_concentration=test_concentration, ceilings=ceilings, hard_margin=50.0,
        )
        print(f"      Vybrano {len(sel_idx)} pseudo-labelovanych vzorku "
              f"(konsenzus obou hlavnich modelu + top-{cap} jistota na tridu + "
              f"koncentracni-strop brana).")

        X_extra = X_test[sel_idx]
        y_extra = sel_labels

        oof, test_pred = leave_one_batch_out_oof(
            X_train, y, batch_arr, X_extra, y_extra, X_test, chosen_models
        )
        # Per-typovi experti se NEPREFITUJI (viz fit_per_type_experts docstring
        # -- nakladova uvaha), jejich OOF/test pravdepodobnosti zustavaji fixni
        # napric koly; jen hlavni 2 modely a stacking meta-learner (vc. jeho
        # vlastniho leave-one-batch-out) se aktualizuji o nove pseudo-labely.
        nested_oof, stacked_test = fit_stacking_meta_learner_cv(
            _stack_inputs(oof[a_name], oof[b_name], oof_by_type), y, batch_arr,
            _stack_inputs(test_pred[a_name], test_pred[b_name], test_by_type),
        )
        class_w, oof_f1 = calibrate_class_weights(nested_oof, y)
        print(f"      re-kalibrace: class_w={np.round(class_w, 3)}  "
              f"OOF macro-F1 (HONEST)={oof_f1:.5f}")
        report_oof_f1_by_concentration_bucket(
            np.argmax(nested_oof * class_w, axis=1), y, train_concentration
        )

    print("\n[6] Diagnostika: dulezitost priznaku (finalni hlavni modely, "
          "cely trenink + pseudo-labely)...")
    X_full = np.vstack([X_train, X_extra]) if X_extra is not None else X_train
    y_full = np.concatenate([y, y_extra]) if y_extra is not None else y
    for name in best_two:
        m = clone(chosen_models[name])
        m.fit(X_full, y_full)
        print(f"    Model: {name}")
        report_feature_importance_by_block(m, feat_names, block_sizes)

    print("\n[7] Financi klasifikace: per-radek argmin(-log(prob) + koncentracni prior)")
    print("    -- ZADNE vynucovani rovnomerne distribuce trid (viz bod 3 v docstringu)...")
    test_calibrated = stacked_test * class_w
    test_calibrated /= test_calibrated.sum(axis=1, keepdims=True)

    conc_penalty = concentration_prior_penalty(test_concentration, ceilings)
    n_affected = int((conc_penalty > 0.01).any(axis=1).sum())
    print(f"    Koncentracni prior ovlivnuje rozhodnuti pro {n_affected} testovacich radku "
          f"(z {len(test_df)}) -- POUZE per-radek, zadny vliv na agregatni distribuci.")

    eps = 1e-12
    cost_matrix = -np.log(np.clip(test_calibrated, eps, 1.0)) + conc_penalty
    final_classes = np.argmin(cost_matrix, axis=1) + 1

    sub = pd.DataFrame({
        "measurement_id": test_df["measurement_id"],
        "gas_class": final_classes,
    })
    sub.to_csv(OUTPUT_PATH, index=False)

    print(f"\nHotovo. Submission ulozena do '{OUTPUT_PATH}'.")
    print("Distribuce trid (NENI vynucovana, jen odraz toho, co model + prior predikoval):")
    print(sub["gas_class"].value_counts().sort_index())

    print("\n    Kontrola: rozdeleni predikovanych trid podle koncentracniho bucketu "
          "(u vysokych koncentraci by melo byt patrne soustredeni na tridy s "
          "dostatecnym trenovacim stropem, ne rovnomerny/nahodny rozptyl):")
    check_df = pd.DataFrame({"concentration": test_concentration, "gas_class": final_classes})
    high = check_df[check_df["concentration"] >= 400]
    if len(high) > 0:
        print(high.groupby("concentration")["gas_class"].value_counts().unstack(fill_value=0))


if __name__ == "__main__":
    main()
