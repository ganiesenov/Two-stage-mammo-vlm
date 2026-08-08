"""
Сборка пакета для чтения радиологами (CPU).

Закрывает R1.10, R2.4, R3.4 — единственные замечания, которые не закрываются вычислениями.

Дизайн заимствован у опубликованных reader study в генерации отчётов:
  * рубрика 5 осей по 4-балльной шкале, двое board-certified радиологов —
    конфигурация, сопоставимая с нашим масштабом (52 выхода);
  * артефакт вместо утверждения: выкладывается заполненный лист и интерфейс,
    а не фраза «мы оценили»;
  * межчитательское согласие считается явно (у нас — Cohen's κ, читателей двое).

Шестая ось добавлена нами и является главной: **пропуск клинически значимой находки**.
В известных нам reader study такой оси нет, а у нас автоматическая метрика показала
35% пропусков у лучшей модели (см. FINDINGS.md). Ось делает чтение радиологами
человеческой валидацией этой метрики, а не просто оценкой качества.

Слепота: источник отчёта скрыт, порядок случаен и различен у каждого читателя,
соответствие «айди ↔ модель» лежит в отдельном файле-ключе, который читателям не выдаётся.

Выход:
  reader_study/images/*.png        — 52 снимка, 1024 px, grayscale
  reader_study/reader_<N>.html     — офлайн-интерфейс, экспорт в JSON/CSV
  reader_study/reader_<N>.csv      — тот же лист для заполнения в Excel
  reader_study/_key.json           — соответствие айди и моделей (НЕ выдавать читателям)
"""
import os, sys, json, csv, random, argparse, html

sys.path.insert(0, os.path.dirname(__file__))
from _common import ROOT, PRED_DIR, IMGS_DIR, find_image

OUT = f"{ROOT}/reader_study"
IMG_OUT = f"{OUT}/images"
LONG_SIDE = 1024

AXES = [
    ("grammar",   "Грамматика и читаемость",
     "Язык, связность, отсутствие бессмысленных формулировок"),
    ("findings",  "Описание находок",
     "Насколько описание соответствует тому, что видно на снимке"),
    ("impression", "Заключение",
     "Корректность итогового вывода"),
    ("birads",    "Категория BI-RADS и рекомендация",
     "Правильность категории и соответствие рекомендации категории"),
    ("overall",   "Общее качество",
     "Пригодность отчёта в целом"),
]
SCALE = [(1, "Неприемлемо"), (2, "Слабо"), (3, "Приемлемо"), (4, "Хорошо")]


def build_images(ids):
    from PIL import Image
    os.makedirs(IMG_OUT, exist_ok=True)
    made = 0
    for iid in ids:
        dst = f"{IMG_OUT}/{iid}.png"
        if os.path.exists(dst):
            continue
        src = find_image(iid)
        if not src:
            print(f"  нет изображения для {iid}")
            continue
        im = Image.open(src).convert("L")
        w, h = im.size
        s = LONG_SIDE / max(w, h)
        if s < 1:
            im = im.resize((int(w * s), int(h * s)), Image.LANCZOS)
        im.save(dst, optimize=True)
        made += 1
    print(f"  изображений подготовлено: {made} (всего в каталоге "
          f"{len(os.listdir(IMG_OUT))})")


def html_page(reader_id, items):
    rows = json.dumps(items, ensure_ascii=False)
    axes = json.dumps([{"k": k, "t": t, "h": h} for k, t, h in AXES], ensure_ascii=False)
    scale = json.dumps(SCALE, ensure_ascii=False)
    return f"""<meta charset="utf-8"><title>Оценка заключений — читатель {reader_id}</title>
<style>
:root{{--bg:#fff;--fg:#111;--mut:#666;--line:#ddd;--acc:#1a56db;--warn:#b91c1c}}
@media(prefers-color-scheme:dark){{:root{{--bg:#111;--fg:#eee;--mut:#999;--line:#333}}}}
*{{box-sizing:border-box}}
body{{margin:0;font:15px/1.5 system-ui,sans-serif;background:var(--bg);color:var(--fg)}}
header{{position:sticky;top:0;background:var(--bg);border-bottom:1px solid var(--line);
padding:10px 16px;display:flex;gap:14px;align-items:center;z-index:9}}
.wrap{{display:grid;grid-template-columns:minmax(320px,1fr) minmax(340px,1fr);gap:18px;padding:16px}}
@media(max-width:900px){{.wrap{{grid-template-columns:1fr}}}}
img{{width:100%;border:1px solid var(--line);border-radius:6px;cursor:zoom-in;background:#000}}
.rep{{border:1px solid var(--line);border-radius:6px;padding:12px;white-space:pre-wrap;
font:14px/1.6 ui-monospace,monospace}}
.ax{{border-top:1px solid var(--line);padding:10px 0}}
.ax b{{display:block}}.ax i{{color:var(--mut);font-style:normal;font-size:13px}}
button{{font:inherit;padding:5px 11px;border:1px solid var(--line);background:transparent;
color:var(--fg);border-radius:6px;cursor:pointer}}
button.on{{background:var(--acc);color:#fff;border-color:var(--acc)}}
button.no.on{{background:var(--warn);border-color:var(--warn)}}
textarea{{width:100%;background:transparent;color:var(--fg);border:1px solid var(--line);
border-radius:6px;padding:8px;font:inherit}}
.prog{{color:var(--mut);font-size:13px;margin-left:auto}}
dialog{{border:none;background:#000;padding:0;max-width:96vw;max-height:96vh}}
dialog img{{max-width:96vw;max-height:96vh;cursor:zoom-out;border:0}}
</style>
<header>
  <b>Читатель {reader_id}</b>
  <button id=prev>←</button><span id=pos></span><button id=next>→</button>
  <button id=exp>Выгрузить результаты</button>
  <span class=prog id=prog></span>
</header>
<div class=wrap>
  <div>
    <img id=img alt="">
    <p id=noimg style="display:none;color:var(--mut);font-size:13px;border:1px dashed
      var(--line);border-radius:6px;padding:12px">
      Изображение не включено в этот выпуск. Снимки DMID распространяются
      правообладателем и доступны в оригинальном релизе набора данных
      (Figshare DOI 10.6084/m9.figshare.24522883). Интерфейс приведён как артефакт
      протокола оценки.</p>
  </div>
  <div>
    <div id=stage1>
      <p style="margin:0 0 6px"><b>Шаг 1. Опишите снимок, не видя заключения.</b></p>
      <p style="color:var(--mut);font-size:13px;margin:0 0 8px">
        Этот ответ фиксируется до показа заключения и служит независимым эталоном.
        Изменить его потом нельзя.</p>
      <div class=ax style="border-top:none">
        <b>Есть ли на снимке клинически значимая находка?</b>
        <div style="margin-top:6px">
          <button data-s1='yes'>Есть</button>
          <button data-s1='no'>Нет</button>
        </div>
        <textarea id=s1txt rows=3 placeholder="Какие именно находки видите (локализация, характер)"></textarea>
      </div>
      <button id=lock>Зафиксировать и показать заключение</button>
    </div>
    <div id=stage2 style="display:none">
      <p style="color:var(--mut);font-size:13px;margin:0 0 8px">
        <b>Шаг 2.</b> Оцените заключение. Источник заключения скрыт намеренно.</p>
      <div class=rep id=gen></div>
      <div style="margin:10px 0"><button id=refbtn>Показать заключение рентгенолога</button></div>
      <div class=rep id=ref style="display:none"></div>
      <div id=form></div>
    </div>
  </div>
</div>
<dialog id=dlg><img id=big></dialog>
<script>
const ITEMS={rows}, AXES={axes}, SCALE={scale}, READER="{reader_id}";
const KEY="reader_"+READER;
let st=JSON.parse(localStorage.getItem(KEY)||"{{}}"), i=0;
const $=id=>document.getElementById(id);
function save(){{localStorage.setItem(KEY,JSON.stringify(st));prog()}}
function prog(){{
  const done=ITEMS.filter(x=>st[x.uid]&&st[x.uid].s1locked&&st[x.uid].overall).length;
  $('prog').textContent=done+" из "+ITEMS.length+" оценено";
}}
function render(){{
  const it=ITEMS[i]; const r=st[it.uid]=st[it.uid]||{{}};
  $('pos').textContent=(i+1)+"/"+ITEMS.length;
  $('img').src="images/"+it.img+".png";
  $('img').onerror=()=>{{$('img').style.display="none";$('noimg').style.display="block";}};
  $('img').onload=()=>{{$('img').style.display="block";$('noimg').style.display="none";}};

  // Шаг 1 — независимая разметка снимка, фиксируется до показа заключения.
  // Без неё «пропуск находки» измерялся бы относительно чужого отчёта, а не снимка.
  $('stage1').style.display = r.s1locked ? "none" : "block";
  $('stage2').style.display = r.s1locked ? "block" : "none";
  $('stage1').querySelectorAll("button[data-s1]").forEach(b=>{{
    b.className = (r.s1finding==b.dataset.s1) ? "on" : "";
    b.onclick=()=>{{r.s1finding=b.dataset.s1; save(); render();}};
  }});
  $('s1txt').value = r.s1what||"";
  $('s1txt').oninput=e=>{{r.s1what=e.target.value; save();}};
  $('lock').onclick=()=>{{
    if(!r.s1finding){{alert("Сначала ответьте, есть ли находка на снимке.");return;}}
    r.s1locked=true; r.s1time=new Date().toISOString(); save(); render();
  }};
  if(!r.s1locked) return;

  $('gen').textContent=it.gen;
  $('ref').textContent=it.ref; $('ref').style.display=r.sawRef?"block":"none";
  $('refbtn').style.display=r.sawRef?"none":"inline-block";
  let h="";
  for(const a of AXES){{
    h+="<div class=ax><b>"+a.t+"</b><i>"+a.h+"</i><div style='margin-top:6px'>";
    for(const s of SCALE) h+="<button data-ax='"+a.k+"' data-v='"+s[0]+"' class='"+
      (r[a.k]==s[0]?"on":"")+"'>"+s[0]+" · "+s[1]+"</button> ";
    h+="</div></div>";
  }}
  h+="<div class=ax><b>Пропущена клинически значимая находка</b>"+
     "<i>Есть ли на снимке находка, которой нет в заключении</i><div style='margin-top:6px'>"+
     "<button data-om='yes' class='no "+(r.omission=="yes"?"on":"")+"'>Да, пропущена</button> "+
     "<button data-om='no' class='"+(r.omission=="no"?"on":"")+"'>Нет</button></div>"+
     "<textarea id=omtxt rows=2 placeholder='Какая именно находка пропущена'>"+
     (r.omissionWhat||"")+"</textarea></div>";
  h+="<div class=ax><b>Комментарий</b><textarea id=cm rows=3>"+(r.comment||"")+"</textarea></div>";
  $('form').innerHTML=h;
  $('form').querySelectorAll("button[data-ax]").forEach(b=>b.onclick=()=>{{
    r[b.dataset.ax]=+b.dataset.v; save(); render();}});
  $('form').querySelectorAll("button[data-om]").forEach(b=>b.onclick=()=>{{
    r.omission=b.dataset.om; save(); render();}});
  $('omtxt').oninput=e=>{{r.omissionWhat=e.target.value; save();}};
  $('cm').oninput=e=>{{r.comment=e.target.value; save();}};
  prog();
}}
$('refbtn').onclick=()=>{{const r=st[ITEMS[i].uid]=st[ITEMS[i].uid]||{{}};r.sawRef=true;save();render();}};
$('prev').onclick=()=>{{if(i>0){{i--;render()}}}};
$('next').onclick=()=>{{if(i<ITEMS.length-1){{i++;render()}}}};
$('img').onclick=()=>{{$('big').src=$('img').src;$('dlg').showModal()}};
$('big').onclick=()=>$('dlg').close();
$('exp').onclick=()=>{{
  const out=ITEMS.map(x=>Object.assign({{uid:x.uid,reader:READER}},st[x.uid]||{{}}));
  const b=new Blob([JSON.stringify(out,null,2)],{{type:"application/json"}});
  const a=document.createElement("a");a.href=URL.createObjectURL(b);
  a.download="reader_"+READER+"_results.json";a.click();
}};
render();
</script>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="two_stage_r64",
                    help="какие предсказания оценивать")
    ap.add_argument("--compare", default=None,
                    help="второй модель для слепого сравнения (удваивает объём)")
    ap.add_argument("--readers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    models = [args.model] + ([args.compare] if args.compare else [])

    pool = []
    for m in models:
        p = f"{PRED_DIR}/{m}__test.json"
        if not os.path.exists(p):
            sys.exit(f"Нет предсказаний {p}")
        for x in json.load(open(p, encoding="utf-8"))["predictions"]:
            pool.append({"img": x["id"], "model": m, "gen": x["hyp"], "ref": x["ref"]})

    rng = random.Random(args.seed)
    for n, it in enumerate(pool):
        it["uid"] = f"R{n:03d}"

    print(f"Пакет для чтения: {len(pool)} позиций, моделей {len(models)}, "
          f"читателей {args.readers}")
    build_images(sorted({it["img"] for it in pool}))

    key = {it["uid"]: {"image": it["img"], "model": it["model"]} for it in pool}
    with open(f"{OUT}/_key.json", "w", encoding="utf-8") as f:
        json.dump({"note": "НЕ выдавать читателям — раскрывает источник",
                   "mapping": key}, f, ensure_ascii=False, indent=2)

    for r in range(1, args.readers + 1):
        items = pool[:]
        rng.shuffle(items)                      # свой порядок у каждого читателя
        blind = [{"uid": it["uid"], "img": it["img"], "gen": it["gen"], "ref": it["ref"]}
                 for it in items]
        with open(f"{OUT}/reader_{r}.html", "w", encoding="utf-8") as f:
            f.write(html_page(r, blind))
        with open(f"{OUT}/reader_{r}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["uid", "s1_finding", "s1_what"] + [k for k, _, _ in AXES] +
                       ["omission", "omission_what", "comment"])
            for it in items:
                w.writerow([it["uid"]] + [""] * (len(AXES) + 5))
        print(f"  → reader_{r}.html / reader_{r}.csv")

    print(f"\n  → {OUT}/_key.json  (не выдавать читателям)")
    print("\nЧто передать радиологам: папку reader_study без _key.json.")
    print("Интерфейс открывается двойным кликом, работает офлайн, сохраняет черновик")
    print("в браузере; по завершении — кнопка «Выгрузить результаты».")
    print("Дальше: 11_reader_agreement.py для κ и сверки с автоматической метрикой пропусков.")


if __name__ == "__main__":
    main()
