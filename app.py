from flask import Flask, request, redirect, url_for, render_template_string, flash
import sqlite3, os
from datetime import date, timedelta
from openpyxl import load_workbook

ROOT=os.path.dirname(os.path.abspath(__file__))
DB=os.path.join(ROOT,'ou_data.db')
app=Flask(__name__); app.secret_key='local-ou-analyzer'

def connect():
    c=sqlite3.connect(DB); c.row_factory=sqlite3.Row
    c.execute("""CREATE TABLE IF NOT EXISTS results(
      id INTEGER PRIMARY KEY, day TEXT, hour TEXT, minute TEXT, value TEXT,
      source TEXT DEFAULT 'manual', updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
      UNIQUE(day,hour,minute))""")
    # Migrasi label jam 24.00–24.50 dari database lama ke 00.00–00.50.
    old_rows=c.execute("SELECT day,minute,value,source,updated_at FROM results WHERE hour='24'").fetchall()
    for r in old_rows:
        existing=c.execute("SELECT id FROM results WHERE day=? AND hour='00' AND minute=?",(r['day'],r['minute'])).fetchone()
        if existing:
            c.execute("UPDATE results SET value=?,source=?,updated_at=? WHERE id=?",
                      (r['value'],r['source'],r['updated_at'],existing['id']))
            c.execute("DELETE FROM results WHERE day=? AND hour='24' AND minute=?",(r['day'],r['minute']))
        else:
            c.execute("UPDATE results SET hour='00' WHERE day=? AND hour='24' AND minute=?",(r['day'],r['minute']))
    c.commit(); return c

def clock_parts(x):
    s=str(x).strip().replace(':','.')
    p=s.split('.')
    if len(p)>=2:
        try:
            h, mi = int(p[0]), int(p[1])
            # Excel pada sumber memakai label 24.00–24.50 untuk interval 00.00–00.50.
            if h == 24:
                h = 0
            if not (0 <= h <= 23 and 0 <= mi <= 59):
                return None
            return f'{h:02d}', f'{mi:02d}'
        except:
            return None
    return None

def import_xlsx(path):
    wb=load_workbook(path,data_only=True); c=connect(); count=0
    months={'jan':1,'feb':2,'mar':3,'apr':4,'mei':5,'may':5,'jun':6,'jul':7,'agu':8,'aug':8,'sep':9,'okt':10,'oct':10,'nov':11,'des':12,'dec':12}
    for sh in wb.worksheets:
        bits=sh.title.lower().strip().split()
        if len(bits)<2: continue
        try: d=int(bits[0]); m=months[bits[1][:3]]; day=f'{date.today().year:04d}-{m:02d}-{d:02d}'
        except: continue
        for row in sh.iter_rows(values_only=True):
            for i in range(0,len(row)-1,2):
                clock,val=row[i],row[i+1]
                if clock is None or str(val).strip().lower() not in ('o','u'): continue
                parts=clock_parts(clock)
                if not parts: continue
                h,mi=parts
                if int(mi) not in (0,10,20,30,40,50): continue
                c.execute("""INSERT INTO results(day,hour,minute,value,source) VALUES(?,?,?,?, 'excel')
                ON CONFLICT(day,hour,minute) DO UPDATE SET value=excluded.value,source='excel',updated_at=CURRENT_TIMESTAMP""",
                (day,h,mi,str(val).strip().lower())); count+=1
    c.commit(); c.close(); return count

def latest_row():
    c=connect(); r=c.execute('SELECT day,hour,minute FROM results ORDER BY day DESC,hour DESC,minute DESC LIMIT 1').fetchone(); c.close(); return r

def majority(values):
    if not values:
        return None, 0, 0, 0
    o=values.count('o'); u=values.count('u'); n=o+u
    return ('o' if o>=u else 'u'), o, u, n

def prior_records(c, target_day, target_hour, target_minute):
    # All records strictly before the target timestamp, in chronological order.
    return c.execute("""SELECT day,hour,minute,value FROM results
        WHERE (day < ?) OR (day = ? AND hour < ?) OR
              (day = ? AND hour = ? AND minute < ?)
        ORDER BY day,hour,minute""",
        (target_day,target_day,target_hour,target_day,target_hour,target_minute)).fetchall()

def probability_o(values, alpha=1.0, beta=1.0):
    """P(O) dengan Beta(alpha,beta) smoothing. O/U diasumsikan label biner."""
    valid=[v for v in values if v in ('o','u')]
    o=valid.count('o'); u=valid.count('u')
    n=o+u
    return ((o+alpha)/(n+alpha+beta), n, o, u)

def predict_methods(c, target_day, target_hour, target_minute):
    prior=prior_records(c,target_day,target_hour,target_minute)
    vals=[r['value'] for r in prior]

    # A: observasi jam+menit sama dari tanggal lebih awal.
    same=[r['value'] for r in prior
          if r['hour']==target_hour and r['minute']==target_minute and r['day']<target_day]
    pa,an,ao,au=probability_o(same)
    a='o' if pa>=0.5 else 'u'

    # B: 20 observasi terakhir sebelum target.
    recent=vals[-20:]
    pb,bn,bo,bu=probability_o(recent)
    b='o' if pb>=0.5 else 'u'

    # C: Markov-1: probabilitas state berikutnya setelah state terakhir yang diketahui.
    pc=None; cn=0; c_pred=None; co=cu=0
    if len(prior)>=2:
        prev=prior[-1]['value']
        transitions=[prior[i]['value'] for i in range(1,len(prior))
                     if prior[i-1]['value']==prev]
        pc,cn,co,cu=probability_o(transitions)
        c_pred='o' if pc>=0.5 else 'u'

    # D: distribusi historis untuk jam target, semua menit dan tanggal sebelumnya.
    hour_values=[r['value'] for r in prior if r['hour']==target_hour]
    pd,dn,do,du=probability_o(hour_values)
    d_pred='o' if pd>=0.5 else 'u'

    # E: distribusi historis untuk label menit target di semua jam sebelumnya.
    minute_values=[r['value'] for r in prior if r['minute']==target_minute]
    pe,en,eo,eu=probability_o(minute_values)
    e_pred='o' if pe>=0.5 else 'u'

    # Ensemble probabilitas: rata-rata probabilitas dari metode yang tersedia.
    # Markov tidak diikutkan bila tidak ada data transisi yang bisa dipakai.
    probs=[pa,pb,pd,pe]
    if pc is not None:
        probs.append(pc)
    p_combined=sum(probs)/len(probs)
    combined='o' if p_combined>=0.5 else 'u'

    return {'a':a,'an':an,'ao':ao,'au':au,'pa':round(pa*100,1),
            'b':b,'bn':bn,'bo':bo,'bu':bu,'pb':round(pb*100,1),
            'c':c_pred,'cn':cn,'co':co,'cu':cu,'pc':round(pc*100,1) if pc is not None else None,
            'd':d_pred,'dn':dn,'do':do,'du':du,'pd':round(pd*100,1),
            'e':e_pred,'en':en,'eo':eo,'eu':eu,'pe':round(pe*100,1),
            'combined':combined,'p_combined':round(p_combined*100,1)}

def forecasts(day,hour):
    c=connect(); out=[]
    for mi in ('00','10','20','30','40','50'):
        m=predict_methods(c,day,hour,mi)
        out.append(dict(minute=mi, **m))
    c.close(); return out

def backtest_scores():
    c=connect()
    rows=c.execute("SELECT day,hour,minute,value FROM results ORDER BY day,hour,minute").fetchall()
    methods={key:{'correct':0,'tested':0} for key in ('a','b','c','d','e','combined')}
    # Walk-forward: predict each result using only rows earlier than its timestamp.
    for r in rows:
        m=predict_methods(c,r['day'],r['hour'],r['minute'])
        for key in methods:
            pred=m.get(key)
            if pred:
                methods[key]['tested']+=1
                if pred==r['value']: methods[key]['correct']+=1
    c.close()
    for key,v in methods.items():
        v['pct']=round(v['correct']/v['tested']*100,1) if v['tested'] else None
    return methods


def probability_stats(group_by):
    """Laporkan probabilitas U dengan Beta(1,1) smoothing; data historis deskriptif."""
    c=connect()
    rows=c.execute(f"SELECT {group_by} AS grp, value, COUNT(*) AS n FROM results GROUP BY {group_by}, value ORDER BY grp").fetchall()
    c.close(); buckets={}
    for r in rows:
        b=buckets.setdefault(r['grp'], {'o':0,'u':0})
        b[r['value']]=r['n']
    out=[]
    for grp,b in sorted(buckets.items()):
        o,u=b['o'],b['u']; n=o+u
        # Laplace/Beta(1,1) estimate: probabilities sum to 1 after adding one pseudo-count each.
        pu=(u+1)/(n+2); po=(o+1)/(n+2)
        out.append({'group':grp,'o':o,'u':u,'n':n,'po':round(po*100,1),'pu':round(pu*100,1)})
    return out

def baseline_scores():
    c=connect(); rows=c.execute("SELECT day,hour,minute,value FROM results ORDER BY day,hour,minute").fetchall()
    correct=tested=0
    seen=[]
    for r in rows:
        if seen:
            vals=[x['value'] for x in seen]
            pred='o' if vals.count('o')>=vals.count('u') else 'u'
            tested+=1; correct+=int(pred==r['value'])
        seen.append(r)
    c.close()
    return {'correct':correct,'tested':tested,'pct':round(correct/tested*100,1) if tested else None}

def next_hour(h): return f'{(int(h)+1)%24:02d}'

@app.route('/')
def index():
    c=connect(); total=c.execute('SELECT COUNT(*) FROM results').fetchone()[0]
    days=c.execute('SELECT COUNT(DISTINCT day) FROM results').fetchone()[0]
    recent=c.execute('SELECT * FROM results ORDER BY day DESC,hour DESC,minute DESC LIMIT 30').fetchall(); c.close()
    latest=latest_row(); target_day=target_hour=None; pred=[]
    if latest:
        target_day=latest['day']; target_hour=next_hour(latest['hour'])
        if int(latest['hour'])==23: target_day=(date.fromisoformat(target_day)+timedelta(days=1)).isoformat()
        pred=forecasts(target_day,target_hour)
    return render_template_string(PAGE,total=total,days=days,recent=recent,latest=latest,target_day=target_day,target_hour=target_hour,pred=pred,scores=backtest_scores(), hour_stats=probability_stats('hour'), minute_stats=probability_stats('minute'), baseline=baseline_scores())

@app.route('/import',methods=['POST'])
def upload():
    f=request.files.get('file')
    if not f or not f.filename.lower().endswith('.xlsx'):
        flash('Pilih file .xlsx.'); return redirect(url_for('index'))
    path=os.path.join(ROOT,'_temp.xlsx'); f.save(path)
    try: flash(f'Import selesai: {import_xlsx(path)} nilai dibaca/disimpan.')
    except Exception as e: flash('Import gagal: '+str(e))
    finally:
        if os.path.exists(path): os.remove(path)
    return redirect(url_for('index'))

@app.route('/save',methods=['POST'])
def save():
    day=request.form.get('day',''); hour=request.form.get('hour','')
    values=[request.form.get('v'+m,'').lower() for m in ('00','10','20','30','40','50')]
    try:
        date.fromisoformat(day)
        if int(hour)<0 or int(hour)>23: raise ValueError()
    except: flash('Tanggal atau jam tidak valid.'); return redirect(url_for('index'))
    if any(v not in ('o','u') for v in values):
        flash('Isi semua enam interval dengan O atau U.'); return redirect(url_for('index'))
    c=connect()
    for mi,v in zip(('00','10','20','30','40','50'),values):
        c.execute("""INSERT INTO results(day,hour,minute,value,source) VALUES(?,?,?,?,'manual')
        ON CONFLICT(day,hour,minute) DO UPDATE SET value=excluded.value,source='manual',updated_at=CURRENT_TIMESTAMP""",(day,hour,mi,v))
    c.commit(); c.close(); flash(f'Aktual {day} jam {hour}.00–{hour}.50 tersimpan.'); return redirect(url_for('index'))

@app.route('/clear',methods=['POST'])
def clear():
    c=connect(); c.execute('DELETE FROM results'); c.commit(); c.close(); flash('Semua data lokal dihapus.'); return redirect(url_for('index'))

PAGE="""
<!doctype html><html lang='id'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>O/U Analyzer</title>
<style>
 :root{--bg:#10151d;--panel:#19222e;--line:#2d3a49;--text:#e8edf4;--muted:#9eacbd;--accent:#28466f}
*{box-sizing:border-box}
html{width:100%;-webkit-text-size-adjust:100%;scroll-behavior:smooth}
body{margin:0;min-width:0;background:var(--bg);color:var(--text);font:14px/1.5 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
main{width:min(100%,1000px);margin:clamp(12px,3vw,25px) auto;padding:0 clamp(10px,3vw,16px)}
header{display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap;padding:5px 0}
h1{font-size:clamp(21px,5vw,28px);margin:0;overflow-wrap:anywhere}h2{font-size:clamp(16px,3.8vw,18px);margin:0 0 13px}
.muted{color:var(--muted)}.card{min-width:0;background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:clamp(12px,3vw,17px);margin:14px 0;overflow:hidden}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,220px),1fr));gap:12px}.stat{font-size:clamp(20px,5vw,26px);font-weight:bold;margin:6px 0;overflow-wrap:anywhere}
input,select,button{max-width:100%;min-width:0;background:#111923;color:var(--text);border:1px solid var(--line);border-radius:8px;padding:10px;font:inherit}
input[type=file]{width:100%;font-size:13px}input[type=date]{width:100%}select{width:100%}button{background:var(--accent);cursor:pointer;min-height:42px;touch-action:manipulation}
button:active{transform:translateY(1px)}label{display:block;color:var(--muted);font-size:12px;margin:0 0 5px}.inputs{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:8px}
.inputs>div{min-width:0}table{width:100%;border-collapse:collapse;min-width:620px}th,td{text-align:left;padding:9px;border-bottom:1px solid var(--line);white-space:nowrap}th{color:var(--muted)}.o{color:#8bc5a5}.u{color:#ffbd8b}.flash{background:#263d32;padding:10px;border-radius:8px;margin:10px 0;overflow-wrap:anywhere}.small{font-size:12px}
.card>div[style*="overflow:auto"]{max-width:100%;overflow-x:auto;-webkit-overflow-scrolling:touch}
form{min-width:0}form p{margin-bottom:0}
@media(max-width:700px){main{margin:10px auto;padding:0 10px}.grid{grid-template-columns:1fr}.inputs{grid-template-columns:repeat(3,minmax(0,1fr))}.card{margin:10px 0}header{align-items:flex-start}header>b{font-size:12px}th,td{padding:8px;font-size:12px}}
@media(max-width:380px){.inputs{grid-template-columns:repeat(2,minmax(0,1fr))}input,select,button{padding:9px;font-size:13px}}
</style></head><body><main>
<header><div><h1>O/U Analyzer</h1><div class='muted'>Analisis historis & input aktual</div></div><b>{{total}} data · {{days}} hari</b></header>
{% for msg in get_flashed_messages() %}<div class='flash'>{{msg}}</div>{% endfor %}
<div class='grid'><section class='card'><h2>Import Excel awal</h2><form action='/import' method='post' enctype='multipart/form-data'><input type='file' name='file' accept='.xlsx' required> <button>Import</button></form><p class='muted small'>Sheet per tanggal, baris berisi pasangan jam dan nilai O/U.</p></section>
<section class='card'><h2>Aktual terakhir</h2>{% if latest %}<div class='stat'>{{latest.day}} · {{latest.hour}}.{{latest.minute}}</div><p class='muted'>Estimasi untuk jam berikutnya.</p>{% else %}<p class='muted'>Import data terlebih dahulu.</p>{% endif %}</section></div>
<section class='card'><h2>Estimasi jam berikutnya</h2>{% if pred %}<p class='muted'>Target {{target_day}} · {{target_hour}}.00–{{target_hour}}.50. Mayoritas historis untuk menit yang sama dari tanggal sebelum target; bukan kepastian.</p>
<div style='overflow:auto'><table><thead><tr><th>Jam</th><th>A: Jam+menit</th><th>B: 20 terakhir</th><th>C: Markov</th><th>D: Per jam</th><th>E: Per menit</th><th>Gabungan</th><th>P(O) gabungan</th></tr></thead><tbody>
{% for r in pred %}<tr><td>{{target_hour}}.{{r.minute}}</td><td class='{{r.a}}'><b>{{r.a|upper if r.a else '—'}}</b> <span class='muted small'>({{r.ao}}O/{{r.au}}U)</span></td><td class='{{r.b}}'><b>{{r.b|upper if r.b else '—'}}</b> <span class='muted small'>({{r.bn}} sampel)</span></td><td class='{{r.c}}'><b>{{r.c|upper if r.c else '—'}}</b> <span class='muted small'>({{r.cn}})</span></td><td class='{{r.d}}'><b>{{r.d|upper}}</b> <span class='muted small'>({{r.dn}})</span></td><td class='{{r.e}}'><b>{{r.e|upper}}</b> <span class='muted small'>({{r.en}})</span></td><td class='{{r.combined}}'><b>{{r.combined|upper}}</b></td><td>{{r.p_combined}}%</td></tr>{% endfor %}</tbody></table></div>
<p class='muted small'>A = historis jam+menit sama; B = 20 data terbaru; C = Markov-1; D = distribusi historis jam target; E = distribusi historis label menit. Gabungan = rata-rata probabilitas metode yang tersedia (Beta(1,1) smoothing), threshold 50%. Bukan kepastian.</p>
<section class='card'><h2>Backtesting (walk-forward)</h2><p class='muted small'>Akurasi dihitung dengan prediksi berurutan hanya memakai data yang tersedia sebelum aktual; hasil awal dengan sampel terbatas turut terhitung.</p>
<div style='overflow:auto'><table><thead><tr><th>Metode</th><th>Cocok</th><th>Diuji</th><th>Akurasi</th></tr></thead><tbody>
{% for key,label in [('a','A: Jam+menit sama'),('b','B: 20 data terakhir'),('c','C: Markov-1'),('d','D: Historis per jam'),('e','E: Historis per menit'),('combined','Gabungan probabilistik')] %}<tr><td>{{label}}</td><td>{{scores[key].correct}}</td><td>{{scores[key].tested}}</td><td>{{scores[key].pct if scores[key].pct is not none else '—'}}%</td></tr>{% endfor %}</tbody></table></div></section>{% else %}<p class='muted'>Belum ada data untuk estimasi.</p>{% endif %}</section>
<section class='card'><h2>Probabilitas historis per jam</h2><p class='muted small'>Persentase deskriptif seluruh data tersimpan. Estimasi memakai smoothing Beta(1,1) agar sampel kecil tidak menghasilkan 0% atau 100%; bukan jaminan hasil berikutnya.</p><div style='overflow:auto'><table><thead><tr><th>Jam</th><th>O (aktual)</th><th>U (aktual)</th><th>N</th><th>P(O) smoothing</th><th>P(U) smoothing</th></tr></thead><tbody>{% for r in hour_stats %}<tr><td>{{r.group}}</td><td>{{r.o}}</td><td>{{r.u}}</td><td>{{r.n}}</td><td>{{r.po}}%</td><td>{{r.pu}}%</td></tr>{% endfor %}</tbody></table></div></section>
<section class='card'><h2>Probabilitas historis per menit</h2><p class='muted small'>Menggabungkan semua jam untuk tiap label menit. Perlu dibaca bersama analisis per jam, bukan sebagai prediksi mandiri.</p><div style='overflow:auto'><table><thead><tr><th>Menit</th><th>O (aktual)</th><th>U (aktual)</th><th>N</th><th>P(O) smoothing</th><th>P(U) smoothing</th></tr></thead><tbody>{% for r in minute_stats %}<tr><td>{{r.group}}</td><td>{{r.o}}</td><td>{{r.u}}</td><td>{{r.n}}</td><td>{{r.po}}%</td><td>{{r.pu}}%</td></tr>{% endfor %}</tbody></table></div></section>
<section class='card'><h2>Baseline mayoritas seluruh data sebelumnya</h2><p class='muted small'>Walk-forward: pada setiap titik, tebak kelas mayoritas dari data yang sudah muncul sebelumnya. Ini pembanding sederhana terhadap metode A/B/C.</p><b>{{baseline.correct}}/{{baseline.tested}} cocok · {{baseline.pct if baseline.pct is not none else '—'}}%</b></section>
<section class='card'><h2>Input aktual per jam</h2><form action='/save' method='post'><div class='grid'><div><label>Tanggal</label><input type='date' name='day' value='{{target_day or ""}}' required></div><div><label>Jam aktual</label><select name='hour'>{% for h in range(24) %}<option value='{{"%02d"|format(h)}}' {% if target_hour==("%02d"|format(h)) %}selected{% endif %}>{{"%02d"|format(h)}}.00–{{"%02d"|format(h)}}.50</option>{% endfor %}</select></div></div>
<div class='inputs'>{% for m in ['00','10','20','30','40','50'] %}<div><label>Menit {{m}}</label><select name='v{{m}}' required><option value=''>—</option><option value='o'>O</option><option value='u'>U</option></select></div>{% endfor %}</div><p><button>Simpan aktual ke database</button></p></form><p class='muted small'>Tanggal+jam yang sudah ada akan diperbarui, bukan diduplikasi.</p></section>
<section class='card'><h2>Data terbaru tersimpan</h2><div style='overflow:auto'><table><thead><tr><th>Tanggal</th><th>Jam</th><th>Aktual</th><th>Sumber</th><th>Diperbarui</th></tr></thead><tbody>{% for r in recent %}<tr><td>{{r.day}}</td><td>{{r.hour}}.{{r.minute}}</td><td class='{{r.value}}'><b>{{r.value|upper}}</b></td><td>{{r.source}}</td><td>{{r.updated_at}}</td></tr>{% endfor %}</tbody></table></div></section>
<form action='/clear' method='post' onsubmit='return confirm("Hapus semua data lokal?")'><button style='background:#522d35'>Hapus seluruh data</button></form>
</main></body></html>
"""
if __name__=='__main__':
    connect().close()
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)
