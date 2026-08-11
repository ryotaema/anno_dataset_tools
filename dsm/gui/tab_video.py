"""動画→画像タブ — mp4 などからフレームを取り出して学習用の画像にする。

取り出す間隔・枚数・範囲を指定し、採用しなかったフレームを予備プール（_rest/）に
残すかどうかまで選べる。出力名は収集スクリプトと同じ命名規則にそろえるので、
抽出後そのまま概要・分割タブでセッション単位に扱える。
"""

import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText

from .. import video as vid_mod
from .common import PathRow

THUMB_COUNT = 5
THUMB_WIDTH = 170
ROW_THUMB_WIDTH = 72        # トラック一覧の各行に出す小さいサムネイル
ROW_THUMB_HEIGHT = 54

# 試算で並べる閾値の候補
SIMILAR_STEPS = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
BLUR_RATIOS = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5]

FILTER_HELP = """品質フィルタは「アノテーションする価値のないフレーム」を抽出から外す機能です。
どちらも閾値を大きくするほど厳しくなり、残る枚数が減ります。

■ 似すぎたフレームを飛ばす（閾値 0〜255・既定 2.0）
  直前に「採用した」フレームと今のフレームを 64x64 の白黒に縮めて比べ、
  1画素あたりの明るさの差の平均を取ります。この値が閾値より小さい
  （＝似ている）フレームは採用しません。弾かれたフレームは比較の基準に
  ならないので、動きが止まっている間はずっと飛ばされます。

    0 に近い … ほぼ同じ絵。定点撮影で何も動いていない区間
    1 〜 3   … わずかな揺れ・照明のちらつき程度の違い
    10 以上  … 構図がはっきり変わっている

  定点タイムラプスや、動きの遅い動画で効きます。同じ絵が大量に入ると、
  アノテーションの手間が増えるうえ train と val に同じ絵が分かれて
  val の成績が実力より良く出ます（情報漏れ）。

■ ブレた画像を弾く（閾値・既定 100）
  画像の輪郭のはっきり具合（Laplacian 分散）を測り、閾値より小さい
  （＝ぼやけている）フレームを採用しません。
  この値の大きさは被写体・解像度・明るさで大きく変わるため、
  絶対的な目安はありません。下の「実測して試算する」で実際の動画の値を
  見てから決めてください（既定の初期値は実測の中央値の半分です）。

  手持ち撮影や、カメラを動かしながらの撮影で効きます。

■ 弾いたフレームの行き先
  「フィルタで弾いた分も _rest/ に残す」を入れておくと、弾いたフレームは
  捨てられずに出力先の _rest/ に入ります。あとから見返して、
  厳しすぎた場合に拾い直せます。
"""


class FilterDialog(tk.Toplevel):
    """品質フィルタの説明と、閾値ごとの残り枚数の試算。"""

    def __init__(self, tab):
        super().__init__(tab)
        self.tab = tab
        self.measure = None
        self.title('品質フィルタの説明と試算')
        self.geometry('900x680')
        self.transient(tab.winfo_toplevel())

        text = ScrolledText(self, height=17, wrap='word')
        text.insert('1.0', FILTER_HELP)
        text.configure(state='disabled')
        text.pack(fill='both', expand=True, padx=10, pady=(10, 6))

        bar = ttk.Frame(self)
        bar.pack(fill='x', padx=10, pady=(0, 4))
        ttk.Button(bar, text='実測して試算する', command=self.run_measure).pack(side='left')
        self.info_var = tk.StringVar(
            value='この動画のフレームを少しだけ読み込んで、閾値ごとに何枚残るかを出します')
        ttk.Label(bar, textvariable=self.info_var).pack(side='left', padx=(10, 0))

        tables = ttk.Frame(self)
        tables.pack(fill='both', expand=True, padx=10, pady=(0, 4))
        self.sim_tree = self._make_table(tables, '似すぎ判定の閾値', '_on_pick_similar')
        self.blur_tree = self._make_table(tables, 'ブレ判定の閾値', '_on_pick_blur')

        self.now_var = tk.StringVar(value='')
        ttk.Label(self, textvariable=self.now_var, foreground='#357').pack(
            anchor='w', padx=14)
        ttk.Label(self, text='※ 表の行をダブルクリックすると、その閾値をタブに反映します',
                  foreground='#555').pack(anchor='w', padx=14, pady=(2, 0))

        btns = ttk.Frame(self)
        btns.pack(fill='x', padx=10, pady=8)
        ttk.Button(btns, text='閉じる', command=self.destroy).pack(side='right')

    def _make_table(self, master, title, handler):
        box = ttk.LabelFrame(master, text=title)
        box.pack(side='left', fill='both', expand=True, padx=(0, 8))
        tree = ttk.Treeview(box, columns=('kept', 'dropped'), height=7)
        tree.heading('#0', text='閾値')
        tree.heading('kept', text='残る枚数（見込み）')
        tree.heading('dropped', text='弾かれる枚数')
        tree.column('#0', width=90, anchor='e', stretch=False)
        tree.column('kept', width=150, anchor='e')
        tree.column('dropped', width=120, anchor='e')
        tree.tag_configure('current', font=('TkDefaultFont', 10, 'bold'))
        tree.pack(fill='both', expand=True, padx=6, pady=6)
        tree.bind('<Double-1>', getattr(self, handler))
        return tree

    # ------------------------------------------------------------------

    def run_measure(self):
        plan = self.tab.current_plan(for_measure=True)
        if plan is None:
            return
        if not plan.plans:
            messagebox.showinfo('動画を読めません', '対象の動画を読み込めませんでした。',
                                parent=self)
            return

        status = self.tab.state.status
        self.info_var.set('実測中...')
        self.update_idletasks()
        self.measure = vid_mod.measure(
            plan, on_progress=lambda d, t: status.step(d, t, '実測中 '))
        status.reset()
        if not self.measure.sampled:
            self.info_var.set('フレームを読み取れませんでした')
            return

        m = self.measure
        self.info_var.set(
            f'サンプル {m.sampled} 枚を実測（抽出候補は全部で {m.candidates:,} 枚）　'
            f'似すぎ度: {vid_mod.stats_text(m.diff_values)}　'
            f'くっきり度: {vid_mod.stats_text(m.blur_values)}')
        self._fill_tables()

    def _fill_tables(self):
        m = self.measure
        self.sim_tree.delete(*self.sim_tree.get_children())
        for th in SIMILAR_STEPS:
            r = vid_mod.simulate(m, similar_thresh=th)
            tag = ('current',) if abs(th - self.tab.similar_th.get()) < 1e-6 else ()
            self.sim_tree.insert('', 'end', text=f'{th:g}', tags=tag,
                                 values=(f'{r["kept"]} / {m.sampled}'
                                         f'（約 {r["kept_total"]:,} 枚）',
                                         f'{r["similar"]}'))

        median = vid_mod.suggest_blur(m) * 2 or 1.0
        self.blur_tree.delete(*self.blur_tree.get_children())
        for ratio in BLUR_RATIOS:
            th = round(median * ratio, 1)
            r = vid_mod.simulate(m, blur_thresh=th)
            self.blur_tree.insert('', 'end', text=f'{th:g}',
                                  values=(f'{r["kept"]} / {m.sampled}'
                                          f'（約 {r["kept_total"]:,} 枚）',
                                          f'{r["blur"]}'))
        self._update_now()

    def _update_now(self):
        if not self.measure:
            return
        tab = self.tab
        r = vid_mod.simulate(
            self.measure,
            blur_thresh=tab.blur_th.get() if tab.blur_var.get() else None,
            similar_thresh=tab.similar_th.get() if tab.similar_var.get() else None)
        parts = []
        parts.append(f'似すぎ {tab.similar_th.get():g}' if tab.similar_var.get()
                     else '似すぎ 使わない')
        parts.append(f'ブレ {tab.blur_th.get():g}' if tab.blur_var.get()
                     else 'ブレ 使わない')
        self.now_var.set(
            f'いまの設定（{" / ".join(parts)}）→ サンプル {self.measure.sampled} 枚中 '
            f'{r["kept"]} 枚が残る見込み（全体で約 {r["kept_total"]:,} 枚。'
            f'似すぎで {r["similar"]} 枚・ブレで {r["blur"]} 枚を除外）')

    def _on_pick_similar(self, _event):
        sel = self.sim_tree.selection()
        if not sel:
            return
        self.tab.similar_th.set(float(self.sim_tree.item(sel[0], 'text')))
        self.tab.similar_var.set(True)
        self.tab._invalidate()
        self._update_now()

    def _on_pick_blur(self, _event):
        sel = self.blur_tree.selection()
        if not sel:
            return
        self.tab.blur_th.set(float(self.blur_tree.item(sel[0], 'text')))
        self.tab.blur_var.set(True)
        self.tab._invalidate()
        self._update_now()


class SourceRow(ttk.Frame):
    """動画・bag のファイル / フォルダのどちらでも選べる入力行。"""

    def __init__(self, master, on_change=None):
        super().__init__(master)
        ttk.Label(self, text='動画 / bag:').pack(side='left')
        self.var = tk.StringVar()
        if on_change:
            self.var.trace_add('write', lambda *a: on_change())
        ttk.Entry(self, textvariable=self.var).pack(side='left', fill='x', expand=True,
                                                    padx=(6, 4))
        ttk.Button(self, text='ファイル...', command=self._pick_file).pack(side='left')
        ttk.Button(self, text='フォルダ...', command=self._pick_dir).pack(side='left', padx=(4, 0))

    def _pick_file(self):
        exts = ' '.join(f'*{e}' for e in sorted(vid_mod.VIDEO_EXTS))
        p = filedialog.askopenfilename(
            title='動画 / bag を選択',
            filetypes=[('動画', exts), ('bag', '*.bag *.db3 *.mcap'), ('すべて', '*.*')])
        if p:
            self.var.set(p)

    def _pick_dir(self):
        p = filedialog.askdirectory(
            title='ROS 2 bag のフォルダ、または動画の入ったフォルダを選択')
        if p:
            self.var.set(p)

    def get(self):
        return self.var.get().strip()


class TrackDialog(tk.Toplevel):
    """ソースの中身（動画/トピック/ストリーム）を見ながら、抽出するものを選ぶ。

    BBox を描画した画像を別トピックで流している場合、名前だけでは判断できない。
    行を選ぶとその絵が出るので、見てから選べる。
    """

    CHECK, UNCHECK = '☑', '☐'

    def __init__(self, tab, paths):
        super().__init__(tab)
        self.tab = tab
        self.title('中身を選ぶ')
        self.geometry('1080x720')
        self.transient(tab.winfo_toplevel())
        self.rows = {}             # iid → {'track', 'checked', 'mod', 'subdir'}
        self._thumbs = []
        self._row_thumbs = []      # 行に置いた PhotoImage の参照（GC されると消える）
        self.result = None

        head = ttk.Frame(self)
        head.pack(fill='x', padx=10, pady=(10, 4))
        self.info_var = tk.StringVar(value='')
        ttk.Label(head, textvariable=self.info_var).pack(side='left')

        # サムネイルを行に表示するので、行の高さを広げておく
        style = ttk.Style(self)
        style.configure('Tracks.Treeview', rowheight=ROW_THUMB_HEIGHT + 8)

        cols = ('kind', 'frames', 'fps', 'start', 'size', 'enc', 'mod', 'dest')
        self.tree = ttk.Treeview(self, columns=cols, height=8, style='Tracks.Treeview')
        self.tree.heading('#0', text='ソース / トラック（ダブルクリックか Space で選択）')
        for c, t, w, anchor in [('kind', '種類', 100, 'center'), ('frames', '枚数', 70, 'e'),
                                ('fps', 'fps', 55, 'e'), ('start', '開始', 70, 'e'),
                                ('size', '解像度', 85, 'center'),
                                ('enc', '形式', 70, 'center'), ('mod', 'モダリティ', 85, 'center'),
                                ('dest', '出力先', 100, 'center')]:
            self.tree.heading(c, text=t)
            self.tree.column(c, width=w, anchor=anchor, stretch=False)
        self.tree.column('#0', width=380, stretch=True)
        self.tree.tag_configure('source', font=('TkDefaultFont', 10, 'bold'))
        self.tree.tag_configure('error', foreground='#b00')
        self.tree.tag_configure('reference', foreground='#357')
        self.tree.pack(fill='both', expand=True, padx=10)
        self.tree.bind('<Double-1>', self._toggle)
        self.tree.bind('<space>', self._toggle)
        self.tree.bind('<<TreeviewSelect>>', lambda e: self._on_select())

        # --- 選択行の設定 ---
        ctl = ttk.LabelFrame(self, text='選択した行の設定')
        ctl.pack(fill='x', padx=10, pady=(6, 4))
        row = ttk.Frame(ctl)
        row.pack(fill='x', padx=8, pady=8)
        ttk.Label(row, text='モダリティ:').pack(side='left')
        self.mod_var = tk.StringVar()
        box = ttk.Combobox(row, textvariable=self.mod_var, width=24, state='readonly',
                           values=[f'{k} — {v}' for k, v in vid_mod.MOD_BASES.items()])
        box.pack(side='left', padx=(4, 14))
        box.bind('<<ComboboxSelected>>', lambda e: self._apply_mod())
        self.dest_var = tk.StringVar(value='main')
        ttk.Radiobutton(row, text='本フォルダ', value='main', variable=self.dest_var,
                        command=self._apply_dest).pack(side='left')
        ttk.Radiobutton(row, text='別フォルダ', value='sub', variable=self.dest_var,
                        command=self._apply_dest).pack(side='left', padx=(6, 0))
        self.subdir_var = tk.StringVar(value='_det')
        ttk.Entry(row, textvariable=self.subdir_var, width=12).pack(side='left', padx=(4, 14))
        ttk.Button(row, text='▲ 上へ（基準にする）',
                   command=lambda: self._move(-1)).pack(side='left')
        ttk.Button(row, text='▼ 下へ', command=lambda: self._move(1)).pack(side='left', padx=(4, 0))

        # --- サムネイル ---
        th = ttk.LabelFrame(self, text='選択した行の画像（BBox が描かれているか確認できます）')
        th.pack(fill='x', padx=10, pady=4)
        self.thumb_frame = ttk.Frame(th)
        self.thumb_frame.pack(fill='x', padx=8, pady=(8, 4))
        bar = ttk.Frame(th)
        bar.pack(fill='x', padx=8, pady=(0, 8))
        self.more_btn = ttk.Button(bar, text='中間・終盤も見る', command=self._more_thumbs)
        self.more_btn.pack(side='left')
        self.thumb_var = tk.StringVar(value='行を選ぶと画像が出ます')
        ttk.Label(bar, textvariable=self.thumb_var, foreground='#555').pack(
            side='left', padx=(8, 0))

        btns = ttk.Frame(self)
        btns.pack(fill='x', padx=10, pady=(0, 10))
        self.summary_var = tk.StringVar(value='')
        ttk.Label(btns, textvariable=self.summary_var, foreground='#357').pack(side='left')
        ttk.Button(btns, text='キャンセル', command=self.destroy).pack(side='right')
        ttk.Button(btns, text='この内容で決定', command=self._ok).pack(side='right', padx=(0, 6))

        self._load(paths)

    # ------------------------------------------------------------------ 読み込み

    def _load(self, paths):
        n_tracks = 0
        for path in paths:
            tracks, error = vid_mod.sources.probe(path)
            kind = vid_mod.sources.detect_kind(path)
            parent = self.tree.insert(
                '', 'end', text=str(path), open=True, tags=('source',),
                values=(vid_mod.KIND_LABELS.get(kind, '不明'), '', '', '', '', '', ''))
            if error:
                self.tree.insert(parent, 'end', text=f'読み込めません: {error}',
                                 tags=('error',), values=('', '', '', '', '', '', ''))
                continue
            if not tracks:
                self.tree.insert(parent, 'end', text='画像のトラックがありません',
                                 tags=('error',), values=('', '', '', '', '', '', ''))
                continue
            first = self.tree.item(parent, 'text')
            started = tracks[0].started_at
            self.tree.item(parent, text=f'{first}   '
                           f'開始 {started:%y-%m-%d %H:%M:%S}（{tracks[0].time_from}）'
                           f' / {tracks[0].duration_text}')
            single = len([t for t in tracks if t.ok]) == 1
            # 描画済みらしいトラックは後ろに置く。先頭のチェックが連番の基準に
            # なるので、既定では生画像が基準になるようにしておく。
            for t in sorted(tracks, key=lambda x: (x.mod_hint == 'det', x.label)):
                # トラックが1本しかないとき（動画など）だけ最初から選んでおく。
                # 複数あるときは、名前から推測して勝手に選ぶと取り違えるので選ばない。
                checked = t.ok and single
                iid = self.tree.insert(
                    parent, 'end',
                    text=f'{self.CHECK if checked else self.UNCHECK} {t.label}',
                    values=(vid_mod.KIND_LABELS.get(t.kind, ''),
                            f'{t.n_frames:,}' + ('(見積)' if t.estimated else ''),
                            f'{t.fps:.1f}',
                            f'+{t.offset:.1f}s' if t.offset else '0.0s',
                            f'{t.width}x{t.height}', t.encoding or '-',
                            t.mod_hint, '本フォルダ'),
                    tags=() if t.ok else ('error',))
                self.rows[iid] = {'track': t, 'checked': checked,
                                  'mod': t.mod_hint, 'subdir': ''}
                n_tracks += 1
                self._set_row_thumb(iid, t)
        self.info_var.set(
            f'トラック {n_tracks} 本。抽出するものにチェックを入れてください'
            '（生画像と BBox 描画済みの両方を選べます。モダリティは名前からの推測なので、'
            '絵を見て直してください）')
        self._update_marks()
        # 最初のトラックを選んで絵を出す
        for iid in self.rows:
            self.tree.selection_set(iid)
            self.tree.focus(iid)
            break

    def _set_row_thumb(self, iid, track):
        """各行に小さいサムネイルを置く（どのトピックが何の絵かを一目で分かるように）。

        ROS 2 bag は必要な位置だけ取り出せるので全行に出す。動画・RealSense は
        先頭から順に読むしかないため、行を選んだときに読み込む。
        """
        if not track.ok or track.kind != 'ros2':
            return
        try:
            mid = vid_mod.sample_indices(track, 3)[1]
            thumbs = vid_mod.track_thumbnails(track, [mid], width=ROW_THUMB_WIDTH)
            if not thumbs:
                return
            img = tk.PhotoImage(data=thumbs[0][1])
        except Exception:
            return                     # 出せなくても選択の妨げにはしない
        self._row_thumbs.append(img)
        self.tree.item(iid, image=img)

    # ------------------------------------------------------------------ 操作

    def _selected(self):
        sel = self.tree.selection()
        for iid in sel:
            if iid in self.rows:
                return iid
        return None

    def _toggle(self, _event=None):
        iid = self._selected()
        if not iid:
            return 'break'
        row = self.rows[iid]
        if not row['track'].ok:
            return 'break'
        row['checked'] = not row['checked']
        self._refresh_row(iid)
        self._update_marks()
        return 'break'

    def _refresh_row(self, iid):
        row = self.rows[iid]
        mark = self.CHECK if row['checked'] else self.UNCHECK
        self.tree.item(iid, text=f'{mark} {row["track"].label}')
        self.tree.set(iid, 'mod', row['mod'])
        self.tree.set(iid, 'dest', row['subdir'] or '本フォルダ')

    def _move(self, delta):
        """基準トラックを選び直すために、同じソース内で行を入れ替える。"""
        iid = self._selected()
        if not iid:
            return
        parent = self.tree.parent(iid)
        index = self.tree.index(iid)
        new = index + delta
        if 0 <= new < len(self.tree.get_children(parent)):
            self.tree.move(iid, parent, new)
            self.tree.selection_set(iid)
            self._update_marks()

    def _apply_mod(self):
        iid = self._selected()
        if iid:
            self.rows[iid]['mod'] = self.mod_var.get().split(' ')[0]
            self._refresh_row(iid)
            self._update_marks()

    def _apply_dest(self):
        iid = self._selected()
        if not iid:
            return
        sub = self.subdir_var.get().strip() if self.dest_var.get() == 'sub' else ''
        self.rows[iid]['subdir'] = sub
        self._refresh_row(iid)

    def _update_marks(self):
        """基準トラック（連番の基準）の表示と、選択数のまとめを更新する。"""
        refs = []
        for parent in self.tree.get_children(''):
            for iid in self.tree.get_children(parent):
                if self.rows.get(iid, {}).get('checked'):
                    refs.append(iid)
                    break
        for iid, row in self.rows.items():
            tags = ('reference',) if iid in refs else ()
            if not row['track'].ok:
                tags = ('error',)
            self.tree.item(iid, tags=tags)
        n = sum(1 for r in self.rows.values() if r['checked'])
        names = [self.rows[i]['track'].label for i in refs]
        msg = (f'{n} 本を抽出します' +
               (f'　連番の基準: {", ".join(names)}（▲上へ で変えられます）' if names else ''))

        # 同じソースでモダリティと出力先が重なると出力名までぶつかる
        dup = {}
        for parent in self.tree.get_children(''):
            for iid in self.tree.get_children(parent):
                row = self.rows.get(iid)
                if row and row['checked']:
                    dup.setdefault((parent, row['mod'], row['subdir']), []).append(
                        row['track'].label)
        clashes = [v for v in dup.values() if len(v) > 1]
        if clashes:
            msg += ('　⚠ 出力名がぶつかります: '
                    + ' / '.join(', '.join(v) for v in clashes)
                    + ' — モダリティか出力先を変えてください')
        self.summary_var.set(msg)

    def _on_select(self):
        iid = self._selected()
        if not iid:
            return
        row = self.rows[iid]
        self.mod_var.set(row['mod'])
        self.dest_var.set('sub' if row['subdir'] else 'main')
        if row['subdir']:
            self.subdir_var.set(row['subdir'])
        # ROS 2 bag は必要な位置だけ取り出せるので3枚まとめて。
        # 動画・RealSense は先頭から順に読むしかないので、まず1枚だけ出す
        track = row['track']
        if track.ok:
            self._show(track, vid_mod.sample_indices(track, 3)
                       if track.kind == 'ros2' else [0])

    def _more_thumbs(self):
        iid = self._selected()
        if iid and self.rows[iid]['track'].ok:
            track = self.rows[iid]['track']
            self._show(track, vid_mod.sample_indices(track, 3))

    def _show(self, track, indices):
        for w in self.thumb_frame.winfo_children():
            w.destroy()
        self._thumbs = []
        self.thumb_var.set('画像を読み込み中...')
        self.update_idletasks()
        try:
            thumbs = vid_mod.track_thumbnails(track, indices)
        except Exception as e:
            self.thumb_var.set(f'画像を読めません: {e}')
            return
        for label, data in thumbs:
            box = ttk.Frame(self.thumb_frame)
            box.pack(side='left', padx=(0, 8))
            try:
                img = tk.PhotoImage(data=data)
            except tk.TclError:
                continue
            self._thumbs.append(img)
            ttk.Label(box, image=img).pack()
            ttk.Label(box, text=label, font=('TkDefaultFont', 8)).pack()
        self.thumb_var.set(
            f'{track.label}（{track.encoding or "-"}）'
            + ('　深度は見やすさのため色を付けています' if track.is_depth else '')
            + ('' if len(indices) > 1 else '　→ 「中間・終盤も見る」で他の位置も確認できます'))

    def _ok(self):
        choices = []
        for parent in self.tree.get_children(''):
            for iid in self.tree.get_children(parent):
                row = self.rows.get(iid)
                if row and row['checked']:
                    choices.append(vid_mod.Choice(track=row['track'], mod=row['mod'],
                                                  subdir=row['subdir']))
        if not choices:
            messagebox.showinfo('選択がありません',
                                '抽出するトラックにチェックを入れてください。', parent=self)
            return
        self.result = choices
        self.tab.set_choices(choices)
        self.destroy()


class VideoTab(ttk.Frame):
    def __init__(self, master, state):
        super().__init__(master)
        self.state = state
        self.plan = None
        self._thumbs = []          # PhotoImage の参照を保持（GCされると消えるため）
        self._last_mode = ''
        self._mode_values = {'sec': 1.0, 'frame': 30.0, 'fps': 2.0, 'count': 100.0}

        self.choices = None        # トラック選択の結果（None なら自動で決める）

        # --- 入力 ---
        src = ttk.LabelFrame(self, text='入力と出力')
        src.pack(fill='x', padx=8, pady=(8, 4))
        self.src_row = SourceRow(src, on_change=self._on_source)
        self.src_row.pack(fill='x', padx=8, pady=(8, 2))
        self.out_row = PathRow(src, '出力先:', on_change=self._invalidate)
        self.out_row.pack(fill='x', padx=8, pady=(2, 4))

        row0 = ttk.Frame(src)
        row0.pack(fill='x', padx=8, pady=(0, 8))
        ttk.Button(row0, text='中身を調べる / トラックを選ぶ...',
                   command=self.open_tracks).pack(side='left')
        self.tracks_var = tk.StringVar(
            value='bag は複数のトピックが入っています。ここで中身を見て選べます')
        ttk.Label(row0, textvariable=self.tracks_var, foreground='#357').pack(
            side='left', padx=(8, 0))

        # --- 取り出す量 ---
        amt = ttk.LabelFrame(self, text='取り出す量')
        amt.pack(fill='x', padx=8, pady=4)

        r1 = ttk.Frame(amt)
        r1.pack(fill='x', padx=8, pady=(8, 2))
        self.mode_var = tk.StringVar(value='sec')
        for key in ('sec', 'frame', 'fps', 'count', 'all'):
            ttk.Radiobutton(r1, text=vid_mod.INTERVAL_MODES[key], value=key,
                            variable=self.mode_var,
                            command=self._on_mode).pack(side='left', padx=(0, 10))
        self.value_var = tk.DoubleVar(value=1.0)
        self.value_var.trace_add('write', lambda *a: self._invalidate())
        self.value_spin = ttk.Spinbox(r1, from_=0.01, to=100000, increment=0.5,
                                      textvariable=self.value_var, width=8)
        self.value_spin.pack(side='left', padx=(8, 4))
        self.unit_var = tk.StringVar(value='秒')
        ttk.Label(r1, textvariable=self.unit_var).pack(side='left')

        r2 = ttk.Frame(amt)
        r2.pack(fill='x', padx=8, pady=(2, 8))
        ttk.Label(r2, text='範囲: 開始').pack(side='left')
        self.start_var = tk.StringVar()
        ttk.Entry(r2, textvariable=self.start_var, width=9).pack(side='left', padx=(4, 4))
        ttk.Label(r2, text='終了').pack(side='left')
        self.end_var = tk.StringVar()
        ttk.Entry(r2, textvariable=self.end_var, width=9).pack(side='left', padx=(4, 4))
        ttk.Label(r2, text='（空欄なら全体・H:MM:SS 可）').pack(side='left')
        ttk.Label(r2, text='　上限: 1動画').pack(side='left')
        self.max_var = tk.IntVar(value=0)
        ttk.Spinbox(r2, from_=0, to=1000000, increment=50, textvariable=self.max_var,
                    width=7).pack(side='left', padx=(4, 4))
        ttk.Label(r2, text='枚　全体').pack(side='left')
        self.max_total_var = tk.IntVar(value=0)
        ttk.Spinbox(r2, from_=0, to=1000000, increment=50,
                    textvariable=self.max_total_var, width=7).pack(side='left', padx=(4, 4))
        ttk.Label(r2, text='枚（0＝制限なし）').pack(side='left')
        for v in (self.start_var, self.end_var, self.max_var, self.max_total_var):
            v.trace_add('write', lambda *a: self._invalidate())

        # --- 名前 ---
        nm = ttk.LabelFrame(self, text='ファイル名（収集スクリプトと同じ命名規則にそろえます）')
        nm.pack(fill='x', padx=8, pady=4)

        r3 = ttk.Frame(nm)
        r3.pack(fill='x', padx=8, pady=(8, 2))
        ttk.Label(r3, text='カメラ:').pack(side='left')
        self.cam_var = tk.StringVar(value='vid')
        ttk.Combobox(r3, textvariable=self.cam_var, width=10,
                     values=vid_mod.CAM_CHOICES).pack(side='left', padx=(4, 12))
        ttk.Label(r3, text='モダリティ:').pack(side='left')
        self.mod_var = tk.StringVar(value='c')
        mod_box = ttk.Combobox(r3, textvariable=self.mod_var, width=24, state='readonly',
                               values=[f'{k} — {v}' for k, v in vid_mod.MOD_BASES.items()])
        mod_box.current(0)
        mod_box.pack(side='left', padx=(4, 12))
        mod_box.bind('<<ComboboxSelected>>', lambda e: self._update_example())
        ttk.Label(r3, text='(bag はトラックごとに選択画面で指定)',
                  foreground='#555').pack(side='left')

        r3b = ttk.Frame(nm)
        r3b.pack(fill='x', padx=8, pady=2)
        self.vmod_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(r3b, text="動画・bag 由来を示す v を付ける（color → cv）",
                        variable=self.vmod_var,
                        command=self._update_example).pack(side='left')
        self.align_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(r3b, text='複数トラックは時刻で対応付けて同じ連番にする',
                        variable=self.align_var,
                        command=self._invalidate).pack(side='left', padx=(16, 0))

        r4 = ttk.Frame(nm)
        r4.pack(fill='x', padx=8, pady=(2, 8))
        ttk.Label(r4, text='日時:').pack(side='left')
        self.time_src = tk.StringVar(value='auto')
        for key, label in [('auto', '自動（bag の記録時刻→ファイル名→更新時刻）'),
                           ('filename', 'ファイル名'),
                           ('mtime', '更新時刻'), ('manual', '手入力')]:
            ttk.Radiobutton(r4, text=label, value=key, variable=self.time_src,
                            command=self._invalidate).pack(side='left', padx=(6, 0))
        self.manual_var = tk.StringVar(value=datetime.now().strftime('%y%m%d_%H%M%S'))
        ttk.Entry(r4, textvariable=self.manual_var, width=14).pack(side='left', padx=(6, 0))
        self.manual_var.trace_add('write', lambda *a: self._invalidate())
        self.example_var = tk.StringVar()
        ttk.Label(nm, textvariable=self.example_var, foreground='#555').pack(
            anchor='w', padx=12, pady=(0, 6))
        self.cam_var.trace_add('write', lambda *a: self._update_example())

        # --- 残り・品質・出力 ---
        rest = ttk.LabelFrame(self, text='採用しなかったフレーム・品質フィルタ・出力')
        rest.pack(fill='x', padx=8, pady=4)

        r5 = ttk.Frame(rest)
        r5.pack(fill='x', padx=8, pady=(8, 2))
        ttk.Label(r5, text='残り:').pack(side='left')
        self.rest_var = tk.StringVar(value='discard')
        for key in ('discard', 'all', 'thinned'):
            ttk.Radiobutton(r5, text=vid_mod.REST_MODES[key], value=key,
                            variable=self.rest_var,
                            command=self._invalidate).pack(side='left', padx=(6, 0))
        self.rest_every = tk.DoubleVar(value=5.0)
        self.rest_every.trace_add('write', lambda *a: self._invalidate())
        ttk.Spinbox(r5, from_=0.1, to=3600, increment=1, textvariable=self.rest_every,
                    width=6).pack(side='left', padx=(6, 2))
        ttk.Label(r5, text='秒ごと').pack(side='left')

        r6 = ttk.Frame(rest)
        r6.pack(fill='x', padx=8, pady=2)
        self.similar_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(r6, text='似すぎたフレームを飛ばす  閾値', variable=self.similar_var,
                        command=self._invalidate).pack(side='left')
        self.similar_th = tk.DoubleVar(value=2.0)
        ttk.Spinbox(r6, from_=0.1, to=255, increment=0.5, textvariable=self.similar_th,
                    width=6).pack(side='left', padx=(4, 2))
        ttk.Label(r6, text='(0〜255・直前の採用フレームとの違い。小さい＝似ている)',
                  foreground='#555').pack(side='left', padx=(0, 14))
        self.blur_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(r6, text='ブレた画像を弾く  閾値', variable=self.blur_var,
                        command=self._invalidate).pack(side='left')
        self.blur_th = tk.DoubleVar(value=100.0)
        ttk.Spinbox(r6, from_=1, to=100000, increment=10, textvariable=self.blur_th,
                    width=8).pack(side='left', padx=(4, 2))
        ttk.Label(r6, text='(輪郭のはっきり具合。小さい＝ぼやけている)',
                  foreground='#555').pack(side='left')

        r6b = ttk.Frame(rest)
        r6b.pack(fill='x', padx=8, pady=(0, 2))
        ttk.Button(r6b, text='フィルタの説明と試算...',
                   command=self.open_filter_help).pack(side='left')
        ttk.Label(r6b, text='← どの値にすると何枚残るかを実測して確かめられます',
                  foreground='#357').pack(side='left', padx=(6, 14))
        self.keep_rejected = tk.BooleanVar(value=True)
        ttk.Checkbutton(r6b, text='フィルタで弾いた分も _rest/ に残す',
                        variable=self.keep_rejected,
                        command=self._invalidate).pack(side='left')
        for v in (self.similar_th, self.blur_th):
            v.trace_add('write', lambda *a: self._invalidate())

        r7 = ttk.Frame(rest)
        r7.pack(fill='x', padx=8, pady=(2, 8))
        ttk.Label(r7, text='形式:').pack(side='left')
        self.ext_var = tk.StringVar(value='.jpg')
        ttk.Combobox(r7, textvariable=self.ext_var, width=6, state='readonly',
                     values=['.jpg', '.png']).pack(side='left', padx=(4, 12))
        ttk.Label(r7, text='JPEG品質:').pack(side='left')
        self.quality_var = tk.IntVar(value=95)
        ttk.Spinbox(r7, from_=50, to=100, textvariable=self.quality_var, width=5
                    ).pack(side='left', padx=(4, 12))
        ttk.Label(r7, text='長辺の上限:').pack(side='left')
        self.max_side = tk.IntVar(value=0)
        ttk.Spinbox(r7, from_=0, to=8000, increment=160, textvariable=self.max_side,
                    width=7).pack(side='left', padx=(4, 4))
        ttk.Label(r7, text='px（0＝原寸）').pack(side='left')
        self.thumb_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(r7, text='プレビューにサムネイルを出す', variable=self.thumb_var
                        ).pack(side='left', padx=(14, 0))

        # --- 実行ボタン（先に下端を確保する） ---
        self.run_btn = ttk.Button(self, text='② 抽出を実行', command=self.run)
        self.run_btn.pack(side='bottom', anchor='w', padx=8, pady=(4, 8))
        self.run_btn.state(['disabled'])

        # --- プレビュー ---
        prev = ttk.LabelFrame(self, text='確認')
        prev.pack(fill='both', expand=True, padx=8, pady=4)
        bar = ttk.Frame(prev)
        bar.pack(fill='x', padx=8, pady=(8, 4))
        ttk.Button(bar, text='① プレビューを作成', command=self.preview).pack(side='left')
        self.summary_var = tk.StringVar(value='動画と出力先を指定してください')
        ttk.Label(bar, textvariable=self.summary_var).pack(side='left', padx=(12, 0))

        cols = ('len', 'fps', 'mod', 'picked', 'rest', 'session', 'example')
        self.tree = ttk.Treeview(prev, columns=cols, height=5)
        self.tree.heading('#0', text='ソース / トラック')
        for c, t, w, anchor in [('len', '長さ', 70, 'e'), ('fps', 'FPS', 55, 'e'),
                                ('mod', 'モダリティ', 80, 'center'),
                                ('picked', '抽出', 70, 'e'),
                                ('rest', '_rest/', 70, 'e'),
                                ('session', 'セッション', 150, 'w'),
                                ('example', '出力名の例', 230, 'w')]:
            self.tree.heading(c, text=t)
            self.tree.column(c, width=w, anchor=anchor, stretch=(c == 'example'))
        self.tree.column('#0', width=230, stretch=False)
        self.tree.tag_configure('reference', foreground='#357')
        self.tree.pack(fill='both', expand=True, padx=8, pady=(0, 4))

        self.thumb_frame = ttk.Frame(prev)
        self.thumb_frame.pack(fill='x', padx=8, pady=(0, 8))

        self._on_mode()

    # ------------------------------------------------------------------ 設定

    def _mod_code(self):
        base = self.mod_var.get().split(' ')[0] or 'c'
        if self.choices:
            base = self.choices[0].mod or base
        return f'{base}v' if self.vmod_var.get() else base

    def _update_example(self, *_):
        cam = self.cam_var.get().strip() or 'vid'
        self.example_var.set(
            f'出力例:  {cam}_260810_143000_00001_{self._mod_code()}{self.ext_var.get()}')
        self._invalidate()

    def _on_mode(self):
        """指定方法を切り替える。値は方法ごとに覚えておく。

        「全体から100枚」から「N秒ごと」に戻したときに 100 秒ごとになってしまうと
        気付きにくい事故になるため、方法ごとに直前の値を保持する。
        """
        mode = self.mode_var.get()
        if self._last_mode and self._last_mode != mode:
            try:
                self._mode_values[self._last_mode] = self.value_var.get()
            except tk.TclError:
                pass
            self.value_var.set(self._mode_values.get(mode, 1.0))
        self._last_mode = mode

        self.unit_var.set({'sec': '秒ごと', 'frame': 'フレームごと', 'fps': '枚/秒',
                           'count': '枚', 'all': ''}.get(mode, ''))
        self.value_spin.state(['disabled'] if mode == 'all' else ['!disabled'])
        self._invalidate()

    def settings(self):
        manual = None
        if self.time_src.get() == 'manual':
            try:
                manual = datetime.strptime(self.manual_var.get().strip(), '%y%m%d_%H%M%S')
            except ValueError:
                raise ValueError('日時の書き方が不正です（YYMMDD_HHMMSS）: '
                                 + self.manual_var.get())
        return vid_mod.Settings(
            mode=self.mode_var.get(), value=self.value_var.get(),
            start=vid_mod.parse_time(self.start_var.get()) or 0.0,
            end=vid_mod.parse_time(self.end_var.get()),
            max_frames=self.max_var.get(), max_total=self.max_total_var.get(),
            cam=self.cam_var.get().strip() or 'vid',
            mod=self.mod_var.get().split(' ')[0] or 'c', video_mod=self.vmod_var.get(),
            time_source=self.time_src.get(), manual_dt=manual,
            ext=self.ext_var.get(), quality=self.quality_var.get(),
            max_side=self.max_side.get(),
            rest=self.rest_var.get(), rest_every=self.rest_every.get(),
            skip_similar=self.similar_var.get(), similar_thresh=self.similar_th.get(),
            skip_blur=self.blur_var.get(), blur_thresh=self.blur_th.get(),
            rest_rejected=self.keep_rejected.get(), align=self.align_var.get())

    def _sources(self):
        """入力欄から動画・bag を探す。"""
        src = self.src_row.get()
        if not src:
            messagebox.showinfo('入力が未指定',
                                '動画か bag のファイル / フォルダを指定してください。')
            return []
        found = vid_mod.find_sources(src)
        if not found:
            messagebox.showinfo(
                '見つかりません',
                f'動画・bag が見つかりません:\n{src}\n\n'
                '対応: 動画（mp4 など） / ROS 2 bag（metadata.yaml のあるフォルダ） / '
                'RealSense の .bag')
        return found

    # ------------------------------------------------------------------ トラック選択

    def _on_source(self, *_):
        """入力が変わったら、前の選択は捨てる（別のソースの選択が残ると混乱する）。"""
        self.choices = None
        self.tracks_var.set('「中身を調べる / トラックを選ぶ」で確認できます')
        self._invalidate()

    def open_tracks(self):
        found = self._sources()
        if not found:
            return
        TrackDialog(self, found)

    def set_choices(self, choices):
        """選択ダイアログの結果を受け取る。"""
        self.choices = choices
        labels = []
        for ch in choices:
            name = ch.track.key or ch.track.path.name
            labels.append(f'{name}[{ch.mod}]' + (f'→{ch.subdir}/' if ch.subdir else ''))
        self.tracks_var.set(f'{len(choices)} 本を選択中: ' + '  '.join(labels[:4])
                            + (f' 他{len(labels) - 4}本' if len(labels) > 4 else ''))
        self._update_example()

    def _plan_targets(self):
        """計画に渡す対象を決める。

        トラックが複数あるソース（bag）は**必ず選択画面を通す**。
        トピック名からの推定で勝手に1本選ぶと、BBox 描画済みの方だけを
        取り出してしまうような取り違えが起きるため。

        Returns:
            (対象, 補足メッセージ, 選択が必要なソース)
        """
        if self.choices:
            return list(self.choices), '', []
        found = self._sources()
        if not found:
            return [], '', []

        targets, need = [], []
        for path in found:
            tracks, error = vid_mod.sources.probe(path)
            if error or not tracks:
                targets.append(path)          # 計画側で「読めません」として扱う
                continue
            ok_tracks = [t for t in tracks if t.ok]
            if len(ok_tracks) > 1:
                need.append(path)             # どれを出すかは名前では決められない
                continue
            track = ok_tracks[0] if ok_tracks else tracks[0]
            targets.append(vid_mod.Choice(
                track=track,
                mod=(self.mod_var.get().split(' ')[0] if track.kind == 'video'
                     else track.mod_hint)))
        return targets, '', need

    def _check_cv2(self):
        try:
            vid_mod.load_cv2()
            return True
        except ImportError as e:
            messagebox.showerror('opencv がありません', str(e))
            return False

    # ------------------------------------------------------------------ 表示

    def _invalidate(self, *_):
        self.plan = None
        self.run_btn.state(['disabled'])

    def refresh(self):
        pass          # 読み込み済みデータセットには依存しないタブ

    def _clear_thumbs(self):
        for w in self.thumb_frame.winfo_children():
            w.destroy()
        self._thumbs = []

    def _show_thumbs(self, plan):
        self._clear_thumbs()
        if not self.thumb_var.get():
            return
        status = self.state.status
        # 選んだフレームまで順に読み進めるので、長い動画ではここが一番待たされる
        scan = max((max(vp.picked) if vp.picked else 0) for vp in plan.plans)
        if scan > 20000:
            self.state.log.write(
                '  サムネイル作成中（長い動画では時間がかかります。'
                '不要なら「プレビューにサムネイルを出す」を外してください）')
        status.message('サムネイル作成中')
        try:
            thumbs = vid_mod.thumbnails(
                plan, count=THUMB_COUNT, width=THUMB_WIDTH,
                on_progress=lambda d, t: status.step(d, t, 'サムネイル '))
        except Exception as e:                      # 表示できなくても本処理は続ける
            self.state.log.write(f'  サムネイルを作成できませんでした: {e}')
            return
        finally:
            self.state.status.reset()
        for label, data in thumbs:
            box = ttk.Frame(self.thumb_frame)
            box.pack(side='left', padx=(0, 8))
            try:
                img = tk.PhotoImage(data=data)
            except tk.TclError:
                return
            self._thumbs.append(img)
            ttk.Label(box, image=img).pack()
            ttk.Label(box, text=label, font=('TkDefaultFont', 8)).pack()

    # ------------------------------------------------------------------ 操作

    def current_plan(self, for_measure=False):
        """いまの設定で抽出計画を作る。作れなければ None（理由はダイアログで出す）。"""
        if not self._check_cv2():
            return None
        targets, note, need = self._plan_targets()
        if need:
            names = '\n'.join(f'  {Path(p).name}' for p in need)
            messagebox.showinfo(
                'どのトラックを出すか選んでください',
                f'次のソースには画像トラック（トピック / ストリーム）が複数あります。\n\n'
                f'{names}\n\n'
                'トピック名だけでは生画像か BBox 描画済みかを判断できないため、\n'
                '選択画面で絵を見て選んでください。')
            self.open_tracks()
            return None
        if not targets:
            return None
        try:
            settings = self.settings()
        except ValueError as e:
            messagebox.showerror('設定エラー', str(e))
            return None
        out = self.out_row.get() or ('.' if for_measure else '')
        if not out:
            messagebox.showinfo('出力先が未指定', '出力フォルダを指定してください。')
            return None
        self._auto_note = note
        self.state.status.message('中身を確認中')
        try:
            return vid_mod.build_plan(targets, out, settings)
        finally:
            self.state.status.reset()

    def open_filter_help(self):
        """品質フィルタの説明と試算のウィンドウを開く。"""
        FilterDialog(self)

    def _existing_count(self, plan):
        """出力先に、今回と同じ名前のファイルが何件あるか（上書きになる件数）。"""
        if not plan.out_dir.is_dir():
            return 0
        n = 0
        for tp in plan.plans:
            base = plan.dir_of(tp)
            if not base.is_dir():
                continue
            for frame in tp.picked:
                if (base / tp.name_of(frame)).exists():
                    n += 1
        return n

    def preview(self):
        self.plan = self.current_plan()
        if self.plan is None:
            return
        settings = self.plan.settings

        # フィルタで弾いた分も _rest/ に入るが、何枚になるかは実際に
        # デコードするまで決まらない。数字だけ見て 0 と誤解しないよう印を付ける。
        filtered = vid_mod.filter_rest_possible(settings)
        mark = ' +α' if filtered else ''

        self.tree.delete(*self.tree.get_children())
        for tp in self.plan.plans:
            track = tp.track
            name = f'{track.path.name}' + (f'  {track.key}' if track.key else '')
            dest = f'{tp.subdir}/' if tp.subdir else ''
            self.tree.insert('', 'end', text=('▶ ' if tp.reference else '   ') + name,
                             tags=('reference',) if tp.reference else (),
                             values=(track.duration_text, f'{track.fps:.1f}',
                                     tp.mod_code,
                                     f'{len(tp.picked):,}{mark}',
                                     f'{len(tp.rest):,}{mark}',
                                     tp.session, dest + tp.first_name))
            if tp.note:
                self.state.log.write(f'  [{name}] {tp.note}')
        for path, reason in self.plan.skipped:
            self.tree.insert('', 'end', text=Path(path).name,
                             values=('', '', '', '', '', '', f'スキップ: {reason}'))

        # 同じ動画を出し直したときは同じ名前になるので黙って上書きされる。
        # 別の動画が同じセッションIDになっている場合もあるため先に知らせる。
        existing = self._existing_count(self.plan)
        if existing:
            self.state.log.write(
                f'  ⚠ 出力先に同じ名前のファイルが {existing:,} 件あります（上書きされます）: '
                f'{self.plan.out_dir}')

        # 同じ名前になるトラックの組み合わせ（生画像と圧縮画像を両方選んだ等）
        for (session, mod, subdir), group in self.plan.conflicts:
            names = ', '.join(tp.track.key or tp.track.path.name for tp in group)
            self.state.log.write(
                f'  ⚠ 出力名が重複します（{mod}{"／" + subdir if subdir else ""}）: {names}')

        total = self.plan.n_picked
        n_tracks = len(self.plan.plans)
        msg = f'{total:,} 枚を出力します'
        if n_tracks > 1:
            msg += f'（トラック {n_tracks} 本の合計）'
        if self.plan.n_rest:
            msg += f'（＋ _rest/ に {self.plan.n_rest:,} 枚）'
        if getattr(self, '_auto_note', ''):
            msg += f'　※ {self._auto_note}'
        if existing:
            msg += f'　⚠ うち {existing:,} 件は既存ファイルの上書き'
        if filtered:
            msg += ('　※ 「+α」は品質フィルタ分。抽出はこれより減り、'
                    '減った分が _rest/ に増えます（枚数は実行時に確定）')
        elif settings.skip_similar or settings.skip_blur:
            msg += '　※ 品質フィルタで実際はこれより少なくなります'
        if self.plan.conflicts:
            msg += (f'　⚠ {len(self.plan.conflicts)} 組のトラックが同じ名前になります'
                    ' — 「中身を調べる」でモダリティか出力先を変えてください（実行できません）')
        elif total:
            msg += '　▶ 下の「② 抽出を実行」で書き出します'
        self.summary_var.set(msg)
        self.run_btn.state(['!disabled'] if (total and self.plan.ok) else ['disabled'])

        self._show_thumbs(self.plan)

    def run(self):
        if not self.plan or not self.plan.ok:
            return
        s = self.plan.settings
        if not messagebox.askyesno(
                '確認',
                f'{self.plan.n_picked:,} 枚を {self.plan.out_dir} に書き出します。\n'
                + (f'採用しなかった {self.plan.n_rest:,} 枚を _rest/ に残します。\n'
                   if self.plan.n_rest else '')
                + ('品質フィルタで弾いた分は抽出から外れ、_rest/ に入ります。\n'
                   if vid_mod.filter_rest_possible(s) else '')
                + '\nトラック:\n'
                + '\n'.join(f'  {tp.mod_code:5} {tp.track.key or tp.track.path.name}'
                            + (f'  → {tp.subdir}/' if tp.subdir else '')
                            for tp in self.plan.plans[:6])
                + ('\n  ...' if len(self.plan.plans) > 6 else '')
                + f'\n\nファイル名: {s.cam}_YYMMDD_HHMMSS_NNNNN_'
                  f'{self._mod_code()}{s.ext}\n\n実行しますか？'):
            return

        self.run_btn.state(['disabled'])
        log, status = self.state.log, self.state.status
        log.write(f'抽出開始 → {self.plan.out_dir}')

        stats = vid_mod.extract(
            self.plan, on_progress=lambda d, t: status.step(d, t, '抽出中 '),
            on_log=log.write)
        status.reset()

        log.write(f'  {stats["picked"]:,} 枚を出力'
                  + (f' / _rest/ に {stats["rest"]:,} 枚' if stats['rest'] else ''))
        if stats['skipped_blur']:
            log.write(f'  ブレのため除外: {stats["skipped_blur"]:,} 枚')
        if stats['skipped_similar']:
            log.write(f'  似すぎたため除外: {stats["skipped_similar"]:,} 枚')
        for path, err in stats['errors'][:10]:
            log.write(f'  [エラー] {path}: {err}')
        log.write(f'  設定を {self.plan.out_dir / "metadata.json"} に保存しました')

        out_dir = self.plan.out_dir
        n = stats['picked']
        self._invalidate()
        if messagebox.askyesno('完了',
                               f'{n:,} 枚を書き出しました。\n\n'
                               'このフォルダをデータセットとして読み込みますか？'):
            ds = self.state.add(out_dir)
            if ds:
                log.write(f'読み込み: {ds.root}  画像={ds.n_images:,}')
                self.state.notify()
