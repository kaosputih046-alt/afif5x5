O/U Analyzer (Flask + SQLite)
Run on Windows:
1) Extract zip and open terminal in folder
2) python -m venv .venv
3) .venv\Scripts\activate
4) pip install -r requirements.txt
5) python app.py
6) Open http://127.0.0.1:5000

Upload your .xlsx workbook. Each sheet should be named like '30 okt', '1 nov'.
Each row should contain alternating time and O/U cells, e.g. 01.00 | o | 01.10 | u.
Import uses the current year on your computer for sheet dates.
The SQLite database is local: ou_data.db.

The next-hour estimate is a historical majority per minute based on earlier dates.
It is descriptive, not guaranteed. Input actuals in six O/U fields to save them.
