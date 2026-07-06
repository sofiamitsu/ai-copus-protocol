import openpyxl
import csv
import sys
import os

INSTRUCTOR_CODES = ["Lec","RtW","FUp","PQ","CQ","AnQ","MG","1o1","D/V","Adm","W","O"]
INSTRUCTOR_START_COL = 2  # Column B
DATA_START_ROW = 4        # Row 4 = window 0-2min

def convert(excel_path, lecture_id, output_csv):
    """
    Reads a filled COPUS instructor-only Excel template
    and converts it to long-format CSV for the validator.
    """
    wb = openpyxl.load_workbook(excel_path, data_only=True)
    ws = wb["COPUS Coding Sheet"]

    rows = []
    for window_idx in range(30):
        row = DATA_START_ROW + window_idx
        codes_present = []

        for i, code in enumerate(INSTRUCTOR_CODES):
            col = INSTRUCTOR_START_COL + i
            val = ws.cell(row=row, column=col).value
            if val and str(val).strip().upper() == "X":
                codes_present.append(code)

        rows.append({
            "lecture_id": lecture_id,
            "window_index": window_idx,
            "window_start": window_idx * 120,
            "window_end": (window_idx + 1) * 120,
            "copus_codes": "|".join(codes_present)
        })

    os.makedirs(os.path.dirname(output_csv) if os.path.dirname(output_csv) else ".", exist_ok=True)

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["lecture_id","window_index","window_start","window_end","copus_codes"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Converted {len(rows)} windows → {output_csv}")
    for r in rows:
        if r["copus_codes"]:
            print(f"  Window {r['window_index']:02d}: {r['copus_codes']}")

if __name__ == "__main__":
    convert(sys.argv[1], sys.argv[2], sys.argv[3])