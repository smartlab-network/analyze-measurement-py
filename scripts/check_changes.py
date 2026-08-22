import os
import csv
import sys


def load_csv_values(path):
    """
    Liest pixel_difference_sum aus CSV.
    """

    values = []

    with open(path, "r") as f:
        reader = csv.DictReader(f)

        for row in reader:
            values.append(
                int(row["pixel_difference_sum"])
            )

    return values



def calculate_alternating_change(values):
    """
    Berechnet die Summe der Änderungen
    innerhalb einer Reihe.

    Beispiel:
    [100,120,130,8000]

    ergibt:
    |120-100| + |130-120| + |8000-130|
    """

    total = 0

    for i in range(1, len(values)):

        total += abs(
            values[i] - values[i-1]
        )

    return total



def analyze_file(path):

    values = load_csv_values(path)


    # 1,3,5,7...
    series_A = values[0::2]

    # 2,4,6,8...
    series_B = values[1::2]


    sum_A = calculate_alternating_change(
        series_A
    )

    sum_B = calculate_alternating_change(
        series_B
    )


    return sum_A, sum_B



def write_results(filename, results):

    with open(filename, "w", newline="") as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "datei",
                "summe_reihe_A_1_3_5",
                "summe_reihe_B_2_4_6"
            ]
        )

        for row in results:

            writer.writerow(row)



def main():

    if len(sys.argv) < 2:

        print(
            "Usage:"
        )

        print(
            "python analyse_csv.py <csv_ordner>"
        )

        sys.exit(1)


    folder = sys.argv[1]


    results = []


    for file in os.listdir(folder):

        if file.lower().endswith(".csv"):

            path = os.path.join(
                folder,
                file
            )


            print(
                "Analysiere:",
                file
            )


            sum_A, sum_B = analyze_file(
                path
            )


            results.append(
                [
                    file,
                    sum_A,
                    sum_B
                ]
            )



    # Gesamttabelle
    write_results(
        os.path.join(
            folder,
            "vergleich_ergebnisse.csv"
        ),
        results
    )



    # Sortierung nach Reihe A
    sorted_A = sorted(
        results,
        key=lambda x: x[1],
        reverse=True
    )

    write_results(
        os.path.join(
            folder,
            "sortiert_nach_A_1_3_5.csv"
        ),
        sorted_A
    )



    # Sortierung nach Reihe B
    sorted_B = sorted(
        results,
        key=lambda x: x[2],
        reverse=True
    )

    write_results(
        os.path.join(
            folder,
            "sortiert_nach_B_2_4_6.csv"
        ),
        sorted_B
    )


    print("\nFertig.")
    print(
        "Ergebnisse gespeichert in:",
        folder
    )



if __name__ == "__main__":
    main()