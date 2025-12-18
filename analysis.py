import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import re

# ---------------------------------------------------------
# 1. DATEN LADEN & BEREINIGEN
# ---------------------------------------------------------

def load_and_clean_data(runs_path, participants_path):
    # Laden der CSV-Dateien
    runs = pd.read_csv("scenario_runs_rows-5.csv")
    participants = pd.read_csv("participants_rows-6.csv")

    # Umbenennen für sauberen Merge
    participants.rename(columns={'id': 'participant_id'}, inplace=True)
    
    # Merge: Füge jedem Run die Teilnehmerdaten hinzu
    df = pd.merge(runs, participants, on='participant_id', how='left')

    #print(df.head(100))

    # Hilfsfunktion für Boolesche Werte (da CSV oft Strings wie "true" enthält)
    def parse_bool(x):
        if isinstance(x, bool): return x
        s = str(x).lower()
        return s in ['true', 't', '1', 'wahr', 'yes']

    # Konvertierung relevanter Spalten
    df['is_biased_ground_truth'] = df['is_biased'].apply(parse_bool) # Ist das Szenario wirklich biased?
    df['is_correct'] = df['is_correct'].apply(parse_bool)            # Hat der User es richtig erkannt?
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df['ai_knowledge'] = pd.to_numeric(df['ai_knowledge'], errors='coerce')

    # Ableitung der User-Antwort (Wichtig für False Positive Rate)
    # Logik: Wenn User richtig lag, entspricht seine Antwort dem Ground Truth. Sonst dem Gegenteil.
    df['user_response_biased'] = df.apply(
        lambda row: row['is_biased_ground_truth'] if row['is_correct'] else not row['is_biased_ground_truth'], 
        axis=1
    )

    # Ableitung der Bias-Stärke aus der scenario_id (z.B. "status-neutral-1" oder "gender-high-2")
    def extract_strength(s_id):
        if pd.isna(s_id): return 'unknown'
        s_id = str(s_id).lower()
        if 'neutral' in s_id: return 'neutral'
        if 'low' in s_id or 'weak' in s_id: return 'low'
        if 'medium' in s_id: return 'medium'
        if 'high' in s_id or 'strong' in s_id: return 'high'
        return 'unknown'

    df['bias_strength'] = df['scenario_id'].apply(extract_strength)
    
    # Kategorie-Mapping bereinigen (falls nötig)
    df['bias_category'] = df['bias_category'].fillna('Unknown')

    return df

# ---------------------------------------------------------
# 2. BERECHNUNG DER ZIELVARIABLEN (User Level)
# ---------------------------------------------------------

def aggregate_user_stats(df):
    # Basis-Metriken pro User
    user_stats = df.groupby('participant_id').agg({
        'is_correct': 'mean',          # Accuracy (Mittelwert der Korrektheit)
        'age': 'first',
        'gender': 'first',
        'ai_knowledge': 'first',
        'id': 'count'                  # Anzahl bearbeiteter Szenarien
    }).rename(columns={'is_correct': 'accuracy', 'id': 'num_scenarios'})

    # Berechnung der False Positive Rate (FPR)
    # FPR = (Anzahl "Bias" gesagt, obwohl "Neutral") / (Anzahl aller neutralen Szenarien)
    neutral_scenarios = df[df['is_biased_ground_truth'] == False]
    
    # Wie oft hat der User hier fälschlicherweise "Bias" (True) vermutet?
    # Wenn is_biased_ground_truth == False, dann bedeutet user_response_biased == True einen Fehler (False Positive)
    if not neutral_scenarios.empty:
        fpr = neutral_scenarios.groupby('participant_id')['user_response_biased'].mean()
        user_stats['fpr'] = fpr
    else:
        user_stats['fpr'] = np.nan

    # Fillna für FPR mit 0, falls User keine neutralen Szenarien falsch hatte (aber welche gesehen hat)
    user_stats['fpr'] = user_stats['fpr'].fillna(0) 
    
    return user_stats.reset_index()

# ---------------------------------------------------------
# 3. STATISTISCHE ANALYSEN
# ---------------------------------------------------------

def run_statistics(user_stats, df):
    print("=== STATISTISCHE AUSWERTUNG ===\n")

    # A) Hypothese: Alter vs. Accuracy
    valid_age = user_stats[['age', 'accuracy']].dropna()
    corr_age, p_age = stats.spearmanr(valid_age['age'], valid_age['accuracy'])
    print(f"1. Alter vs. Erkennungsleistung (Spearman):")
    print(f"   Korrelation (rho): {corr_age:.3f}")
    print(f"   p-Wert: {p_age:.4f}")
    if p_age < 0.05: print("   -> Signifikanter Zusammenhang!")
    print("-" * 30)

    # B) Hypothese: KI-Wissen vs. Accuracy
    valid_know = user_stats[['ai_knowledge', 'accuracy']].dropna()
    corr_know, p_know = stats.spearmanr(valid_know['ai_knowledge'], valid_know['accuracy'])
    print(f"2. KI-Wissen vs. Erkennungsleistung (Spearman):")
    print(f"   Korrelation (rho): {corr_know:.3f}")
    print(f"   p-Wert: {p_know:.4f}")
    print("-" * 30)

    # C) Gruppe: Geschlecht (Mann vs. Frau)
    # Filterung auf Strings, um Varianten wie "Männlich", "männlich", "Mann" abzufangen
    males = user_stats[user_stats['gender'].str.lower().str.contains('männlich|mann', na=False)]['accuracy']
    females = user_stats[user_stats['gender'].str.lower().str.contains('weiblich|frau', na=False)]['accuracy']
    
    print(f"3. Geschlechtervergleich (Mann-Whitney-U):")
    print(f"   Männer (n={len(males)}) Ø Accuracy: {males.mean():.2%}")
    print(f"   Frauen (n={len(females)}) Ø Accuracy: {females.mean():.2%}")
    
    if len(males) > 0 and len(females) > 0:
        u_stat, p_gender = stats.mannwhitneyu(males, females, alternative='two-sided')
        print(f"   p-Wert: {p_gender:.4f}")
    else:
        print("   -> Nicht genügend Daten für Test.")
    print("-" * 30)

    # D) Deskriptiv: Accuracy pro Bias-Kategorie
    # Wir filtern neutrale Szenarien hier raus, da wir wissen wollen: "Wenn Bias da ist, wird er erkannt?"
    biased_only = df[df['is_biased_ground_truth'] == True]
    acc_by_cat = biased_only.groupby('bias_category')['is_correct'].agg(['mean', 'count'])
    print("4. Erkennungsrate pro Bias-Kategorie (Sensitivity):")
    print(acc_by_cat)
    print("\n")

    return acc_by_cat

# ---------------------------------------------------------
# 4. QUALITATIVE ANALYSE (Keywords)
# ---------------------------------------------------------

def analyze_text_reasoning(df):
    print("=== QUALITATIVE ANALYSE (Begründungen) ===")
    
    # Einfache Keyword-Listen (erweiterbar)
    keywords = {
        'Wortwahl/Ton': ['wortwahl', 'ton', 'formulierung', 'sprache', 'ausdruck', 'aggressiv'],
        'Stereotype': ['stereotyp', 'klischee', 'rolle', 'typisch', 'bild', 'vorurteil'],
        'Ungleichbehandlung': ['ungerecht', 'unfair', 'diskriminier', 'rassis', 'sexis', 'bevorzug'],
        'Intuition/Gefühl': ['gefühl', 'bauch', 'glaube', 'wirkt', 'scheint', 'subjektiv'],
        'Vergleich': ['vergleich', 'unterschied', 'besser als', 'schlechter als']
    }

    # Zählen
    results = {k: 0 for k in keywords}
    
    def scan_text(text):
        if not isinstance(text, str): return
        text = text.lower()
        for cat, words in keywords.items():
            if any(w in text for w in words):
                results[cat] += 1

    df['reasoning'].apply(scan_text)
    
    print("Häufigkeit genannter Aspekte in Freitexten:")
    for k, v in results.items():
        print(f"   {k}: {v}")
    
    return results

# ---------------------------------------------------------
# 5. VISUALISIERUNG
# ---------------------------------------------------------

def plot_results(user_stats, acc_by_cat, keyword_counts):
    sns.set_theme(style="whitegrid")
    
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3)

    # Plot 1: Alter vs Accuracy
    ax1 = fig.add_subplot(gs[0, 0])
    sns.regplot(x='age', y='accuracy', data=user_stats, ax=ax1, color='#2ecc71', scatter_kws={'alpha':0.6})
    ax1.set_title('Korrelation: Alter & Erkennungsleistung')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Alter')
    ax1.set_ylim(0, 1.05)

    # Plot 2: KI Wissen (Boxplot)
    ax2 = fig.add_subplot(gs[0, 1])
    sns.boxplot(x='ai_knowledge', y='accuracy', data=user_stats, ax=ax2, palette='Blues')
    ax2.set_title('Einfluss von KI-Vorwissen')
    ax2.set_xlabel('Selbsteinschätzung (1-5)')
    ax2.set_ylabel('Accuracy')

    # Plot 3: FPR Distribution (Histogramm)
    ax3 = fig.add_subplot(gs[0, 2])
    sns.histplot(user_stats['fpr'], bins=5, kde=True, ax=ax3, color='#e74c3c')
    ax3.set_title('Verteilung der False Positive Rate')
    ax3.set_xlabel('FPR (Bias in neutralen Szenarien vermutet)')

    # Plot 4: Kategorien Performance
    ax4 = fig.add_subplot(gs[1, 0:2]) # Breiter
    # Sortieren für bessere Optik
    acc_by_cat_sorted = acc_by_cat.sort_values(by='mean', ascending=False)
    sns.barplot(x=acc_by_cat_sorted.index, y=acc_by_cat_sorted['mean'], ax=ax4, palette='viridis')
    ax4.set_title('Erkennungsrate nach Bias-Kategorie')
    ax4.set_ylabel('Anteil korrekt erkannt')
    ax4.set_ylim(0, 1)
    for i, v in enumerate(acc_by_cat_sorted['mean']):
        ax4.text(i, v + 0.02, f"{v:.1%}", ha='center', fontweight='bold')

    # Plot 5: Qualitative Keywords
    ax5 = fig.add_subplot(gs[1, 2])
    keys = list(keyword_counts.keys())
    vals = list(keyword_counts.values())
    sns.barplot(x=vals, y=keys, ax=ax5, palette='magma', orient='h')
    ax5.set_title('Qualitative Analyse: Argumentationsmuster')
    ax5.set_xlabel('Anzahl Nennungen')

    plt.tight_layout()
    plt.savefig('auswertung_ergebnisse.png', dpi=300)
    print("\nGrafik gespeichert als 'auswertung_ergebnisse.png'")
    plt.show()

# ---------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------

if __name__ == "__main__":
    # Dateinamen anpassen, falls nötig
    FILE_RUNS = 'scenario_runs_rows-5.csv'
    FILE_PARTICIPANTS = 'participants_rows-6.csv'

    try:
        # 1. Daten laden
        df_full = load_and_clean_data(FILE_RUNS, FILE_PARTICIPANTS)
        
        # Export des gemergten DataFrames
        output_file = 'merged_data_export.csv'
        df_full.to_csv(output_file, index=False, encoding='utf-8')
        print(f"\n✅ Gemergtes DataFrame exportiert als '{output_file}'")
        print(f"   Anzahl Zeilen: {len(df_full)}, Anzahl Spalten: {len(df_full.columns)}\n")
        
        # 2. Aggregieren
        df_users = aggregate_user_stats(df_full)
        
        # 3. Statistik & Ausgabe
        category_stats = run_statistics(df_users, df_full)
        
        # 4. Text Analyse
        text_stats = analyze_text_reasoning(df_full)
        
        # 5. Plotten
        plot_results(df_users, category_stats, text_stats)
        
    except FileNotFoundError as e:
        print(f"FEHLER: Datei nicht gefunden. {e}")
    except Exception as e:
        print(f"Ein unerwarteter Fehler ist aufgetreten: {e}")