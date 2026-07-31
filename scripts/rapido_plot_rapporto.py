import matplotlib.pyplot as plt
import pandas as pd


def calcola_e_plotta_rapporto(file1, file2):
    # Carica i due file CSV (con riga di intestazione)
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    # Estrae la seconda colonna (indice 1) da entrambi i file
    colonna_df1 = df1.iloc[:, 1]
    colonna_df2 = df2.iloc[:, 1]

    # Allinea i dati in base alla lunghezza minore per evitare errori
    lunghezza_minima = min(len(colonna_df1), len(colonna_df2))
    dati1 = colonna_df1.iloc[:lunghezza_minima]
    dati2 = colonna_df2.iloc[:lunghezza_minima]

    # Calcola il rapporto coppia per coppia (file1 / file2)
    rapporto = dati1 / dati2

    # Crea il grafico
    plt.figure(figsize=(10, 6))
    plt.plot(rapporto, marker=".", linestyle="-", color="b", label="Rapporto")

    # Configura i testi del grafico
    plt.title("Rapporto tra le seconde colonne dei file CSV")
    plt.xlabel("Indice della riga")
    plt.ylabel("Valore del rapporto (File 1 / File 2)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()

    # Mostra il grafico a schermo
    plt.show()


calcola_e_plotta_rapporto('D:/Goya/flatfield_analyses_120kVp/first_flatfield_counts.csv', 'D:/Goya/flatfield_analyses_120kVp/last_flatfield_counts.csv')
