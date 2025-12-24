# 📊 Bias-Erkennungsanalyse

Python-Skript zur statistischen Analyse von Bias-Erkennungsdaten aus einer Studie zur KI-Bias-Wahrnehmung.

## 📋 Überblick

Das Skript `analysis.py` analysiert Daten aus einer Studie, in der Teilnehmer*innen KI-Szenarien auf Bias bewertet haben. Es führt statistische Analysen durch, erstellt Visualisierungen und generiert detaillierte Berichte.

### Hauptfunktionen

- **Datenverarbeitung**: Merge von Szenario-Runs und Teilnehmerdaten
- **Statistische Analysen**: 
  - Korrelationen (Alter, KI-Wissen, AI-Attitude, AI-Reliance vs. Erkennungsleistung)
  - Geschlechtervergleiche
  - Erkennungsraten pro Bias-Kategorie
  - False Positive Rate (FPR) Analyse
- **Demografische Statistiken**: Alter, KI-Wissen, Einstellungen, Geschlechterverteilung
- **Szenario-Performance**: Detaillierte Analyse pro Szenario-ID
- **Visualisierungen**: Automatische Generierung von Grafiken
- **Qualitative Analyse**: Keyword-basierte Analyse der Begründungen

## 🚀 Installation & Setup

### 1. Virtuelle Umgebung erstellen

**Wichtig**: Es wird empfohlen, eine virtuelle Umgebung zu verwenden, um Abhängigkeiten sauber zu verwalten.

#### macOS/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

#### Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

### 2. Dependencies installieren

Nach Aktivierung der virtuellen Umgebung:

```bash
pip install -r requirements.txt
```

### 3. Dateien vorbereiten

Stelle sicher, dass folgende CSV-Dateien im Projektverzeichnis vorhanden sind:
- `scenario_runs_rows-10.csv` (oder entsprechend angepasst)
- `participants_rows-11.csv` (oder entsprechend angepasst)

Die Dateinamen können in `analysis.py` im `__main__`-Block angepasst werden:
```python
FILE_RUNS = 'scenario_runs_rows-10.csv'
FILE_PARTICIPANTS = 'participants_rows-11.csv'
```

## ▶️ Ausführung

### Skript starten

```bash
python analysis.py
```

### Was passiert beim Ausführen?

1. **Daten laden**: Die CSV-Dateien werden geladen und gemergt
2. **Export**: Ein gemergtes DataFrame wird als `merged_data_export.csv` exportiert
3. **Statistische Auswertung**: 
   - Filterung nach Mindestanzahl Szenarien (Standard: ≥3)
   - Demografische Statistiken
   - Korrelationsanalysen
   - Geschlechtervergleiche
   - Bias-Kategorie-Analysen
4. **Szenario-Performance**: Analyse pro Szenario-ID
5. **Qualitative Analyse**: Keyword-Analyse der Begründungen
6. **Visualisierung**: Grafiken werden generiert und als `auswertung_ergebnisse.png` gespeichert

## 📊 Ausgabe

### Konsolen-Ausgabe

Das Skript gibt folgende Statistiken aus:

- **Grundstatistiken**: Anzahl Teilnehmer, durchschnittliche Runs
- **Demografische Angaben**: Alter, KI-Wissen, AI-Attitude, AI-Reliance, Geschlechterverteilung
- **Korrelationen**: 
  - Alter vs. Erkennungsleistung
  - KI-Wissen vs. Erkennungsleistung
  - AI-Attitude vs. Erkennungsleistung
  - AI-Reliance vs. Erkennungsleistung
- **Geschlechtervergleich**: Mann-Whitney-U Test
- **Bias-Kategorien**: Erkennungsrate pro Kategorie (age, ethnicity, gender, status)
- **Szenario-Performance**: Detaillierte Tabelle pro Szenario-ID
- **Qualitative Analyse**: Häufigkeit von Argumentationsmustern

### Generierte Dateien

- `merged_data_export.csv`: Vollständiges gemergtes Dataset
- `auswertung_ergebnisse.png`: Visualisierungen (6 Grafiken)

## ⚙️ Konfiguration

### Mindestanzahl Szenarien

In der Funktion `run_statistics()` kann die Mindestanzahl an Szenarien pro Teilnehmer angepasst werden:

```python
MIN_SCENARIOS = 3  # Standard: 3
```

Teilnehmer mit weniger Szenarien werden von den statistischen Analysen ausgeschlossen.

### Ground Truth

Die Ground Truth wird automatisch aus der `scenario_id` abgeleitet:
- Alle Szenarien sind **biased** (`is_biased_ground_truth = True`)
- Außer: `status-neutral-1` ist **nicht biased** (`is_biased_ground_truth = False`)

## 📦 Dependencies

- **pandas**: Datenmanipulation und -analyse
- **numpy**: Numerische Berechnungen
- **matplotlib**: Visualisierungen
- **seaborn**: Statistische Grafiken
- **scipy**: Statistische Tests (Spearman, Mann-Whitney-U)

## 📝 Beispiel-Ausgabe

```
=== STATISTISCHE AUSWERTUNG ===

Gesamt Teilnehmer mit Accuracy-Daten: 31
Filterung: Mindestanzahl Szenarien = 3
Teilnehmer mit mindestens 3 Szenarien: 28 (ausgeschlossen: 3)

=== DEMOGRAFISCHE ANGABEN ===

Durchschnittsalter: 39.4 Jahre (n=28, Min: 17, Max: 84)
Durchschnittliches KI-Wissen (1-5): 2.81 (n=31)
...

=== Szenario-Performance (pro scenario_id) ===
                           bias_category  is_biased  n_runs  n_participants  accuracy
scenario_id                                                                          
...
```

## 🔧 Troubleshooting

### Fehler: "ModuleNotFoundError"
- Stelle sicher, dass die virtuelle Umgebung aktiviert ist
- Installiere alle Dependencies: `pip install -r requirements.txt`

### Fehler: "FileNotFoundError"
- Überprüfe, ob die CSV-Dateien im richtigen Verzeichnis liegen
- Passe die Dateinamen in `analysis.py` an

### Fehler bei der Ausführung
- Überprüfe, ob Python 3.8+ verwendet wird: `python --version`
- Stelle sicher, dass alle CSV-Dateien korrekt formatiert sind

## 👤 Autor

Alexander Gnwuni - Bachelor Thesis Project

## 📄 Lizenz

Dieses Projekt ist für akademische und Bildungszwecke verfügbar.
