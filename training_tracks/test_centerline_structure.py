#!/usr/bin/env python3
"""
Prueba unitaria para verificar la estructura correcta del archivo centerline.csv
"""

import os
import csv
import sys


def test_centerline_structure():
    """
    Verifica que los archivos centerline.csv tengan la estructura correcta:
    - Header: # x_m,y_m,w_tr_right_m,w_tr_left_m
    - 4 columnas de datos numéricos
    """
    print("🧪 Testing centerline.csv structure...")

    tracks_dir = "./tracks"
    expected_header = ['# x_m', 'y_m', 'w_tr_right_m', 'w_tr_left_m']

    if not os.path.exists(tracks_dir):
        print(f"❌ Tracks directory '{tracks_dir}' not found")
        return False

    tracks_tested = 0
    all_passed = True

    for track_name in os.listdir(tracks_dir):
        track_path = os.path.join(tracks_dir, track_name)

        if os.path.isdir(track_path):
            centerline_file = os.path.join(track_path, f"{track_name}_centerline.csv")

            if os.path.exists(centerline_file):
                print(f"  🔍 Testing {track_name}_centerline.csv...")

                try:
                    with open(centerline_file, 'r') as f:
                        reader = csv.reader(f)

                        # Verificar header
                        header = next(reader)
                        if header != expected_header:
                            print(f"    ❌ Wrong header: {header}")
                            print(f"    Expected: {expected_header}")
                            all_passed = False
                            continue

                        # Verificar que todas las filas tengan 4 columnas numéricas
                        row_count = 0
                        for row_num, row in enumerate(reader, start=2):
                            if len(row) != 4:
                                print(f"    ❌ Row {row_num}: Expected 4 columns, got {len(row)}")
                                all_passed = False
                                continue

                            # Verificar que todos los valores sean numéricos
                            try:
                                [float(val) for val in row]
                            except ValueError as e:
                                print(f"    ❌ Row {row_num}: Non-numeric value: {e}")
                                all_passed = False
                                continue

                            row_count += 1

                        if row_count == 0:
                            print("    ❌ No data rows found")
                            all_passed = False
                        else:
                            print(f"    ✅ {row_count} valid data rows")

                        tracks_tested += 1

                except Exception as e:
                    print(f"    ❌ Error reading file: {e}")
                    all_passed = False
            else:
                print(f"  ⚠️  {track_name}: centerline.csv not found")

    if tracks_tested == 0:
        print("❌ No centerline files found to test")
        return False

    if all_passed:
        print(f"✅ All {tracks_tested} centerline files have correct structure!")
        return True
    else:
        print("❌ Some centerline files have incorrect structure")
        return False


def test_specific_track_format(track_name="oval_small"):
    """Prueba específica para un track individual"""
    print(f"\n🎯 Detailed test for {track_name}...")

    centerline_file = f"./tracks/{track_name}/{track_name}_centerline.csv"

    if not os.path.exists(centerline_file):
        print(f"❌ File not found: {centerline_file}")
        return False

    with open(centerline_file, 'r') as f:
        lines = f.readlines()

    print(f"📄 File has {len(lines)} lines")

    # Mostrar las primeras 5 líneas
    print("📋 First 5 lines:")
    for i, line in enumerate(lines[:5]):
        print(f"  {i+1}: {line.strip()}")

    # Verificar formato usando csv.reader
    with open(centerline_file, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        first_data_row = next(reader)

        print(f"🏷️  Header: {header}")
        print(f"📊 First data row: {first_data_row}")
        print(f"🔢 Data types: {[type(float(val)).__name__ for val in first_data_row]}")

    return True


if __name__ == "__main__":
    success = test_centerline_structure()
    test_specific_track_format("oval_small")

    if not success:
        sys.exit(1)

    print("\n🎉 All tests passed!")
