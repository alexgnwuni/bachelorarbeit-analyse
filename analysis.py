import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import re
import json

# ---------------------------------------------------------
# 1. DATEN LADEN & BEREINIGEN
# ---------------------------------------------------------

def load_and_clean_data(runs_path, participants_path):
    # Laden der CSV-Dateien
    runs = pd.read_csv("scenario_runs_rows-10.csv")
    participants = pd.read_csv("participants_rows-11.csv")

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
    df['is_correct'] = df['is_correct'].apply(parse_bool)                  # Hat der User korrekt bewertet?
    df['user_response_biased'] = df['is_biased'].apply(parse_bool)        # Nutzerantwort: biased ja/nein
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df['ai_knowledge'] = pd.to_numeric(df['ai_knowledge'], errors='coerce')
    df['ai_attitude'] = pd.to_numeric(df['ai_attitude'], errors='coerce')
    df['ai_reliance'] = pd.to_numeric(df['ai_reliance'], errors='coerce')

    # Ableitung der Ground Truth: Alle Szenarien sind biased, außer status-neutral-1
    df['is_biased_ground_truth'] = df['scenario_id'] != 'status-neutral-1'

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
        'ai_attitude': 'first',
        'ai_reliance': 'first',
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
    
    # Berechnung der False Positive Rate für falsche Kategorien (FPR Category)
    # FPR Category = (Anzahl biasede Szenarien mit falscher Kategorie) / (Anzahl aller biaseden Szenarien, bei denen Bias erkannt wurde)
    # WICHTIG: Nur Szenarien, die eine Kategorie erfordern:
    # - gender-biased-1 und age-biased-1: guessed_bias_category muss angegeben werden
    # - explore-*: bias_strength_ratings JSON wird ausgewertet
    # - status-neutral-1 und compas-biased-1: werden ausgeschlossen (keine Kategorie erforderlich)
    
    biased_scenarios = df[df['is_biased_ground_truth'] == True]
    
    if not biased_scenarios.empty:
        # Nur Szenarien, bei denen der Nutzer Bias erkannt hat
        biased_detected = biased_scenarios[biased_scenarios['user_response_biased'] == True].copy()
        
        if not biased_detected.empty:
            # Filtere Szenarien, die keine Kategorie erfordern (status-neutral-1, compas-biased-1)
            # Diese haben immer null in guessed_bias_category und werden ausgeschlossen
            scenarios_requiring_category = biased_detected[
                ~biased_detected['scenario_id'].isin(['status-neutral-1', 'compas-biased-1'])
            ]
            
            if not scenarios_requiring_category.empty:
                def has_wrong_category(row):
                    scenario_id = str(row.get('scenario_id', '')).lower()
                    actual_category = str(row.get('bias_category', '')).strip().lower()
                    
                    # Für explore-* Szenarien: bias_strength_ratings JSON auswerten
                    if scenario_id.startswith('explore-'):
                        ratings_str = row.get('bias_strength_ratings', '')
                        if pd.isna(ratings_str) or str(ratings_str).strip() == '' or str(ratings_str).strip().lower() == 'null':
                            return True  # Keine Ratings = falsch
                        
                        try:
                            # Parse JSON
                            if isinstance(ratings_str, str):
                                ratings = json.loads(ratings_str)
                            else:
                                ratings = ratings_str
                            
                            if not isinstance(ratings, dict):
                                return True  # Ungültiges Format
                            
                            # Prüfe, ob die richtige Kategorie den höchsten Wert hat (oder mindestens gleich hoch)
                            if actual_category not in ratings:
                                return True  # Richtige Kategorie fehlt
                            
                            actual_value = ratings.get(actual_category, 0)
                            max_value = max(ratings.values()) if ratings else 0
                            
                            # Falsch, wenn die richtige Kategorie nicht den höchsten Wert hat
                            return actual_value < max_value
                            
                        except (json.JSONDecodeError, ValueError, TypeError):
                            return True  # Fehler beim Parsen = falsch
                    
                    # Für gender-biased-1 und age-biased-1: guessed_bias_category muss mit bias_category übereinstimmen
                    else:
                        guessed_val = row.get('guessed_bias_category', np.nan)
                        
                        if pd.isna(guessed_val) or str(guessed_val).strip() == '' or str(guessed_val).strip().lower() == 'null':
                            return True  # Keine Kategorie geraten = falsch
                        
                        guessed = str(guessed_val).strip().lower()
                        return guessed != actual_category
                
                scenarios_requiring_category['wrong_category'] = scenarios_requiring_category.apply(has_wrong_category, axis=1)
                fpr_category = scenarios_requiring_category.groupby('participant_id')['wrong_category'].mean()
                user_stats['fpr_category'] = fpr_category
            else:
                user_stats['fpr_category'] = np.nan
        else:
            user_stats['fpr_category'] = np.nan
    else:
        user_stats['fpr_category'] = np.nan
    
    # Fillna für FPR Category mit 0, falls User keine biaseden Szenarien mit falscher Kategorie hatte
    user_stats['fpr_category'] = user_stats['fpr_category'].fillna(0)
    
    return user_stats.reset_index()

# ---------------------------------------------------------
# 3. STATISTISCHE ANALYSEN
# ---------------------------------------------------------

def run_statistics(user_stats, df):
    print("=== STATISTISCHE AUSWERTUNG ===\n")
    
    # Konstante für Mindestanzahl Szenarien
    MIN_SCENARIOS = 3
    
    # Gesamtanzahl aller Teilnehmer (mit Accuracy)
    total_with_accuracy = len(user_stats[user_stats['accuracy'].notna()])
    print(f"Gesamt Teilnehmer mit Accuracy-Daten: {total_with_accuracy}")
    
    # Durchschnittliche Runs pro Teilnehmer VOR Filterung (alle mit mindestens 1 Szenario)
    avg_runs_before = user_stats['num_scenarios'].mean()
    print(f"Durchschnittliche Runs pro Teilnehmer (vor Filterung, alle mit ≥1 Szenario): {avg_runs_before:.2f}")
    
    # Filterung: Nur Teilnehmer mit mindestens MIN_SCENARIOS Szenarien
    user_stats = user_stats[user_stats['num_scenarios'] >= MIN_SCENARIOS]
    total_after_filter = len(user_stats[user_stats['accuracy'].notna()])
    excluded_count = total_with_accuracy - total_after_filter
    
    print(f"Filterung: Mindestanzahl Szenarien = {MIN_SCENARIOS}")
    print(f"Teilnehmer mit mindestens {MIN_SCENARIOS} Szenarien: {total_after_filter} (ausgeschlossen: {excluded_count})")
    
    # Durchschnittliche Runs pro Teilnehmer (nach Filterung)
    avg_runs_after = user_stats['num_scenarios'].mean()
    print(f"Durchschnittliche Runs pro Teilnehmer (nach Filterung, ≥{MIN_SCENARIOS} Szenarien): {avg_runs_after:.2f}")
    
    # Durchschnittliche Genauigkeit (nach Filterung)
    avg_accuracy = user_stats['accuracy'].mean()
    print(f"Durchschnittliche Genauigkeit (nach Filterung, ≥{MIN_SCENARIOS} Szenarien): {avg_accuracy:.2%}")
    
    # Durchschnittliche False Positive Rate (Bias in neutralen Szenarien vermutet)
    avg_fpr = user_stats['fpr'].mean()
    print(f"Durchschnittliche False Positive Rate (FPR): {avg_fpr:.2%} (Bias in neutralen Szenarien vermutet)")
    
    # Durchschnittliche False Positive Rate für falsche Kategorien
    avg_fpr_category = user_stats['fpr_category'].mean()
    print(f"Durchschnittliche FPR für falsche Kategorien: {avg_fpr_category:.2%} (Bias erkannt, aber falsche Kategorie gewählt)\n")
    
    # Demografische Statistiken
    print("=== DEMOGRAFISCHE ANGABEN ===\n")
    
    # Durchschnittsalter
    ages = user_stats['age'].dropna()
    if len(ages) > 0:
        avg_age = ages.mean()
        min_age = ages.min()
        max_age = ages.max()
        print(f"Durchschnittsalter: {avg_age:.1f} Jahre (n={len(ages)}, Min: {min_age:.0f}, Max: {max_age:.0f})")
    else:
        print("Durchschnittsalter: Keine Daten verfügbar")
    
    # Durchschnittliches KI-Wissen
    ai_knowledge = user_stats['ai_knowledge'].dropna()
    if len(ai_knowledge) > 0:
        avg_ai_knowledge = ai_knowledge.mean()
        print(f"Durchschnittliches KI-Wissen (1-5): {avg_ai_knowledge:.2f} (n={len(ai_knowledge)})")
    else:
        print("Durchschnittliches KI-Wissen: Keine Daten verfügbar")
    
    # Durchschnittliche AI-Attitude
    ai_attitude = user_stats['ai_attitude'].dropna()
    if len(ai_attitude) > 0:
        avg_ai_attitude = ai_attitude.mean()
        print(f"Durchschnittliche AI-Attitude (1-5): {avg_ai_attitude:.2f} (n={len(ai_attitude)})")
    else:
        print("Durchschnittliche AI-Attitude: Keine Daten verfügbar")
    
    # Durchschnittliche AI-Reliance
    ai_reliance = user_stats['ai_reliance'].dropna()
    if len(ai_reliance) > 0:
        avg_ai_reliance = ai_reliance.mean()
        print(f"Durchschnittliche AI-Reliance (1-5): {avg_ai_reliance:.2f} (n={len(ai_reliance)})")
    else:
        print("Durchschnittliche AI-Reliance: Keine Daten verfügbar")
    
    # Geschlechterverteilung
    print("\nGeschlechterverteilung:")
    # Erstelle eine Kopie für sichere String-Operationen
    gender_lower = user_stats['gender'].astype(str).str.lower()
    males = user_stats[gender_lower.str.contains('männlich|mann', na=False, regex=True)]
    females = user_stats[gender_lower.str.contains('weiblich|frau', na=False, regex=True)]
    no_gender = user_stats[user_stats['gender'].isna()]
    
    print(f"   Männlich: {len(males)} ({len(males)/total_after_filter*100:.1f}%)")
    print(f"   Weiblich: {len(females)} ({len(females)/total_after_filter*100:.1f}%)")
    if len(no_gender) > 0:
        print(f"   Keine Angabe: {len(no_gender)} ({len(no_gender)/total_after_filter*100:.1f}%)")
    
    print("\n" + "=" * 30 + "\n")

    # A) Hypothese: Alter vs. Accuracy
    # Verwendet ALLE Teilnehmer mit Alter und Accuracy (auch ohne Geschlecht)
    valid_age = user_stats[['age', 'accuracy']].dropna()
    corr_age, p_age = stats.spearmanr(valid_age['age'], valid_age['accuracy'])
    print(f"1. Alter vs. Erkennungsleistung (Spearman):")
    print(f"   Teilnehmer (n={len(valid_age)}): Alle mit Alter und Accuracy")
    print(f"   Korrelation (rho): {corr_age:.3f}")
    print(f"   p-Wert: {p_age:.4f}")
    if p_age < 0.05: print("   -> Signifikanter Zusammenhang!")
    print("-" * 30)

    # B) Hypothese: KI-Wissen vs. Accuracy
    # Verwendet ALLE Teilnehmer mit KI-Wissen und Accuracy (auch ohne Geschlecht)
    valid_know = user_stats[['ai_knowledge', 'accuracy']].dropna()
    corr_know, p_know = stats.spearmanr(valid_know['ai_knowledge'], valid_know['accuracy'])
    print(f"2. KI-Wissen vs. Erkennungsleistung (Spearman):")
    print(f"   Teilnehmer (n={len(valid_know)}): Alle mit KI-Wissen und Accuracy")
    print(f"   Korrelation (rho): {corr_know:.3f}")
    print(f"   p-Wert: {p_know:.4f}")
    print("-" * 30)

    # B2) Hypothese: AI-Attitude vs. Accuracy
    # Verwendet ALLE Teilnehmer mit AI-Attitude und Accuracy
    valid_attitude = user_stats[['ai_attitude', 'accuracy']].dropna()
    corr_attitude, p_attitude = stats.spearmanr(valid_attitude['ai_attitude'], valid_attitude['accuracy'])
    print(f"2a. AI-Attitude vs. Erkennungsleistung (Spearman):")
    print(f"   Teilnehmer (n={len(valid_attitude)}): Alle mit AI-Attitude und Accuracy")
    print(f"   Korrelation (rho): {corr_attitude:.3f}")
    print(f"   p-Wert: {p_attitude:.4f}")
    if p_attitude < 0.05: print("   -> Signifikanter Zusammenhang!")
    print("-" * 30)

    # B3) Hypothese: AI-Reliance vs. Accuracy
    # Verwendet ALLE Teilnehmer mit AI-Reliance und Accuracy
    valid_reliance = user_stats[['ai_reliance', 'accuracy']].dropna()
    corr_reliance, p_reliance = stats.spearmanr(valid_reliance['ai_reliance'], valid_reliance['accuracy'])
    print(f"2b. AI-Reliance vs. Erkennungsleistung (Spearman):")
    print(f"   Teilnehmer (n={len(valid_reliance)}): Alle mit AI-Reliance und Accuracy")
    print(f"   Korrelation (rho): {corr_reliance:.3f}")
    print(f"   p-Wert: {p_reliance:.4f}")
    if p_reliance < 0.05: print("   -> Signifikanter Zusammenhang!")
    print("-" * 30)

    # C) Gruppe: Geschlecht (Mann vs. Frau)
    # Verwendet NUR Teilnehmer MIT Geschlecht (13 von 19)
    # Filterung auf Strings, um Varianten wie "Männlich", "männlich", "Mann" abzufangen
    males = user_stats[user_stats['gender'].str.lower().str.contains('männlich|mann', na=False)]['accuracy']
    females = user_stats[user_stats['gender'].str.lower().str.contains('weiblich|frau', na=False)]['accuracy']
    
    print(f"3. Geschlechtervergleich (Mann-Whitney-U):")
    print(f"   Teilnehmer (n={len(males) + len(females)}): Nur mit Geschlecht angegeben")
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

    return acc_by_cat, user_stats

# ---------------------------------------------------------
# 4. SZENARIO-ANALYSE (pro scenario_id)
# ---------------------------------------------------------

def analyze_scenario_performance(df_runs):
    """
    Beschreibt, wie gut einzelne Szenarien (scenario_id) erkannt werden.
    Es werden alle Runs berücksichtigt, unabhängig davon, wie viele Szenarien
    ein*e Teilnehmer*in insgesamt bearbeitet hat.
    """
    df_filtered_runs = df_runs.copy()

    if df_filtered_runs.empty:
        print("=== Szenario-Performance (pro scenario_id) ===")
        print("Keine Daten nach Filterung verfügbar.\n")
        return None, df_filtered_runs

    scenario_overview = (
        df_filtered_runs
        .groupby('scenario_id')
        .agg(
            bias_category=('bias_category', 'first'),
            is_biased=('is_biased_ground_truth', 'first'),
            n_runs=('id', 'count'),
            n_participants=('participant_id', 'nunique'),
            accuracy=('is_correct', 'mean')
        )
        .sort_values('accuracy', ascending=False)
    )

    print("=== Szenario-Performance (pro scenario_id) ===")
    print(scenario_overview)
    print("\n")

    return scenario_overview, df_filtered_runs


def print_scenario_details(df_filtered_runs, scenario_id):
    """
    Detailansicht: Zeigt für ein konkretes scenario_id, wie einzelne Teilnehmer
    entschieden haben (inkl. einiger demografischer Variablen).
    Diese Funktion wird nicht automatisch im Main ausgeführt, sondern kann
    bei Bedarf manuell aufgerufen werden.
    """
    subset = df_filtered_runs[df_filtered_runs['scenario_id'] == scenario_id]

    print(f"=== Details für Szenario {scenario_id} ===")
    if subset.empty:
        print("Keine Daten für dieses Szenario (nach Filterung).\n")
        return

    cols = [
        'participant_id',
        'is_correct',
        'is_biased_ground_truth',
        'bias_category',
        'age',
        'gender',
        'ai_knowledge',
        'ai_attitude',
        'ai_reliance'
    ]
    existing_cols = [c for c in cols if c in subset.columns]
    print(subset[existing_cols])
    print("\n")


# ---------------------------------------------------------
# 5. QUALITATIVE ANALYSE (Keywords)
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
    FILE_RUNS = 'scenario_runs_rows-10.csv'
    FILE_PARTICIPANTS = 'participants_rows-11.csv'

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
        
        # 3. Statistik & Ausgabe (gibt gefilterte user_stats zurück)
        category_stats, df_users_filtered = run_statistics(df_users, df_full)

        # 3b. Szenario-Analyse (pro scenario_id, nutzt ALLE Runs)
        scenario_overview, df_filtered_runs = analyze_scenario_performance(df_full)
        # Für Detailansichten einzelner Szenarien kann man z.B. aufrufen:
        # print_scenario_details(df_filtered_runs, \"compas-biased-1\")
        
        # 4. Text Analyse
        text_stats = analyze_text_reasoning(df_full)
        
        # 5. Plotten (verwendet gefilterte user_stats)
        plot_results(df_users_filtered, category_stats, text_stats)
        
    except FileNotFoundError as e:
        print(f"FEHLER: Datei nicht gefunden. {e}")
    except Exception as e:
        print(f"Ein unerwarteter Fehler ist aufgetreten: {e}")