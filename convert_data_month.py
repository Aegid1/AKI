import os


company_name = "Wirtschaft"
year = "2024"
path = f"data/news/{company_name}"

german_months = {
    "Januar": "01", "Februar": "02", "März": "03", "April": "04", "Mai": "05", "Juni": "06",
    "Juli": "07", "August": "08", "September": "09", "Oktober": "10", "November": "11", "Dezember": "12"
}

# Iteration durch alle Dateien im angegebenen Pfad
for filename in os.listdir(path):
    # Überspringe keine Dateien oder Verzeichnisse
    if not os.path.isfile(os.path.join(path, filename)):
        continue

    # Suche nach Monatsnamen im Dateinamen und ersetze sie durch die Nummer
    for month_name, month_number in german_months.items():
        if month_name in filename:
            new_filename = filename.replace(month_name, month_number)
            old_path = os.path.join(path, filename)
            new_path = os.path.join(path, new_filename)

            # Datei umbenennen
            os.rename(old_path, new_path)
            print(f"Renamed: {old_path} -> {new_path}")
            break  # Nach der Umbenennung kann der nächste Dateiname geprüft werden

