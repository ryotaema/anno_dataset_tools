"""動画→画像タブ — mp4 などからフレームを取り出して学習用の画像にする。

取り出す間隔・枚数・範囲を指定し、採用しなかったフレームを予備プール（_rest/）に
残すかどうかまで選べる。出力名は収集スクリプトと同じ命名規則にそろえるので、
抽出後そのまま概要・分割タブでセッション単位に扱える。
"""

import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import ttk, filedialog, messagebox

from .. import video as vid_mod
from .common import PathRow

THUMB_COUNT = 5
THUMB_WIDTH = 170


class SourceRow(ttk.Frame):
    """動画ファイル / フォルダのどちらでも選べる入力行。"""

    def __init__(self, master, on_change=None):
        super().__init__(master)
        ttk.Label(self, text='動画:').pack(side='left')
        self.var = tk.StringVar()
        if on_change:
            self.var.trace_add('write', lambda *a: on_change())
        ttk.Entry(self, textvariable=self.var).pack(side='left', fill='x', expand=True,
                                                    padx=(6, 4))
        ttk.Button(self, text='ファイル...', command=self._pick_file).pack(side='left')
        ttk.Button(self, text='フォルダ...', command=self._pick_dir).pack(side='left', padx=(4, 0))

    def _pick_file(self):
        exts = ' '.join(f'*{e}' for e in sorted(vid_mod.VIDEO_EXTS))
        p = filedialog.askopenfilename(title='動画を選択',
                                       filetypes=[('動画', exts), ('すべて', '*.*')])
        if p:
            self.var.set(p)

    def _pick_dir(self):
        p = filedialog.askdirectory(title='動画の入ったフォルダを選択')
        if p:
            self.var.set(p)

    def get(self):
        return self.var.get().strip()


class VideoTab(ttk.Frame):
    def __init__(self, master, state):
        super().__init__(master)
        self.state = state
        self.plan = None
        self._thumbs = []          # PhotoImage の参照を保持（GCされると消えるため）

        # --- 入力 ---
        src = ttk.LabelFrame(self, text='入力と出力')
        src.pack(fill='x', padx=8, pady=(8, 4))
        self.src_row = SourceRow(src, on_change=self._invalidate)
        self.src_row.pack(fill='x', padx=8, pady=(8, 2))
        self.out_row = PathRow(src, '出力先:', on_change=self._invalidate)
        self.out_row.pack(fill='x', padx=8, pady=(2, 8))

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
        self.vmod_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(r3, text="動画由来を示す v を付ける", variable=self.vmod_var,
                        command=self._update_example).pack(side='left')

        r4 = ttk.Frame(nm)
        r4.pack(fill='x', padx=8, pady=(2, 8))
        ttk.Label(r4, text='日時:').pack(side='left')
        self.time_src = tk.StringVar(value='auto')
        for key, label in [('auto', '自動（ファイル名→更新時刻）'), ('filename', 'ファイル名'),
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
                    width=6).pack(side='left', padx=(4, 14))
        self.blur_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(r6, text='ブレた画像を弾く  閾値', variable=self.blur_var,
                        command=self._invalidate).pack(side='left')
        self.blur_th = tk.DoubleVar(value=100.0)
        ttk.Spinbox(r6, from_=1, to=100000, increment=10, textvariable=self.blur_th,
                    width=8).pack(side='left', padx=(4, 4))
        ttk.Button(r6, text='推奨値を測る', command=self.measure_blur).pack(side='left')
        self.keep_rejected = tk.BooleanVar(value=True)
        ttk.Checkbutton(r6, text='弾いた分も _rest/ に残す', variable=self.keep_rejected
                        ).pack(side='left', padx=(14, 0))

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

        cols = ('len', 'fps', 'size', 'picked', 'rest', 'session', 'example')
        self.tree = ttk.Treeview(prev, columns=cols, height=5)
        self.tree.heading('#0', text='動画')
        for c, t, w, anchor in [('len', '長さ', 70, 'e'), ('fps', 'FPS', 55, 'e'),
                                ('size', '解像度', 90, 'center'), ('picked', '抽出', 70, 'e'),
                                ('rest', '_rest/', 70, 'e'),
                                ('session', 'セッション', 150, 'w'),
                                ('example', '出力名の例', 230, 'w')]:
            self.tree.heading(c, text=t)
            self.tree.column(c, width=w, anchor=anchor, stretch=(c == 'example'))
        self.tree.column('#0', width=200, stretch=False)
        self.tree.pack(fill='both', expand=True, padx=8, pady=(0, 4))

        self.thumb_frame = ttk.Frame(prev)
        self.thumb_frame.pack(fill='x', padx=8, pady=(0, 8))

        self._on_mode()

    # ------------------------------------------------------------------ 設定

    def _mod_code(self):
        base = self.mod_var.get().split(' ')[0] or 'c'
        return f'{base}v' if self.vmod_var.get() else base

    def _update_example(self, *_):
        cam = self.cam_var.get().strip() or 'vid'
        self.example_var.set(
            f'出力例:  {cam}_260810_143000_00001_{self._mod_code()}{self.ext_var.get()}')
        self._invalidate()

    def _on_mode(self):
        mode = self.mode_var.get()
        self.unit_var.set({'sec': '秒', 'frame': 'フレーム', 'fps': '枚/秒',
                           'count': '枚', 'all': ''}.get(mode, ''))
        self.value_spin.state(['disabled'] if mode == 'all' else ['!disabled'])
        if mode == 'count' and self.value_var.get() < 2:
            self.value_var.set(100)
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
            rest_rejected=self.keep_rejected.get())

    def _videos(self):
        src = self.src_row.get()
        if not src:
            messagebox.showinfo('動画が未指定', '動画ファイルかフォルダを指定してください。')
            return []
        videos = vid_mod.find_videos(src)
        if not videos:
            messagebox.showinfo('動画がありません',
                                f'動画ファイルが見つかりません:\n{src}')
        return videos

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
        self.state.status.message('サムネイル作成中')
        try:
            thumbs = vid_mod.thumbnails(plan, count=THUMB_COUNT, width=THUMB_WIDTH)
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

    def measure_blur(self):
        """候補フレームを少し実測して、ブレ閾値の推奨値を入れる。"""
        if not self._check_cv2():
            return
        videos = self._videos()
        if not videos:
            return
        try:
            plan = vid_mod.build_plan(videos, self.out_row.get() or '.', self.settings())
        except ValueError as e:
            messagebox.showerror('設定エラー', str(e))
            return
        if not plan.plans:
            messagebox.showinfo('動画を読めません', '対象の動画を読み込めませんでした。')
            return

        status = self.state.status
        q = vid_mod.sample_quality(
            plan, on_progress=lambda d, t: status.step(d, t, '測定中 '))
        status.reset()
        if not q['blur']:
            messagebox.showinfo('測定できません', 'フレームを読み取れませんでした。')
            return
        self.blur_th.set(q['suggest_blur'])
        self.blur_var.set(True)
        lo, hi = min(q['blur']), max(q['blur'])
        self.state.log.write(
            f'ブレ判定の実測: 最小 {lo:.0f} / 最大 {hi:.0f} → 閾値を {q["suggest_blur"]:.0f} '
            f'にしました（小さいほどブレている）')
        self._invalidate()

    def preview(self):
        if not self._check_cv2():
            return
        videos = self._videos()
        if not videos:
            return
        out = self.out_row.get()
        if not out:
            messagebox.showinfo('出力先が未指定', '出力フォルダを指定してください。')
            return
        try:
            settings = self.settings()
        except ValueError as e:
            messagebox.showerror('設定エラー', str(e))
            return

        self.state.status.message('動画を確認中')
        self.plan = vid_mod.build_plan(videos, out, settings)
        self.state.status.reset()

        self.tree.delete(*self.tree.get_children())
        mod, ext = settings.mod_code, settings.ext
        for vp in self.plan.plans:
            info = vp.info
            self.tree.insert('', 'end', text=info.path.name,
                             values=(info.duration_text, f'{info.fps:.1f}',
                                     f'{info.width}x{info.height}',
                                     f'{len(vp.picked):,}', f'{len(vp.rest):,}',
                                     vp.session, vp.first_name(mod, ext)))
            if vp.note:
                self.state.log.write(f'  [{info.path.name}] {vp.note}')
        for path, reason in self.plan.skipped:
            self.tree.insert('', 'end', text=path.name,
                             values=('', '', '', '', '', '', f'スキップ: {reason}'))

        total = self.plan.n_picked
        msg = f'{total:,} 枚を出力します'
        if self.plan.n_rest:
            msg += f'（＋ _rest/ に {self.plan.n_rest:,} 枚）'
        if settings.skip_similar or settings.skip_blur:
            msg += '　※ 品質フィルタで実際はこれより少なくなります'
        if total:
            msg += '　▶ 下の「② 抽出を実行」で書き出します'
        self.summary_var.set(msg)
        self.run_btn.state(['!disabled'] if total else ['disabled'])

        self._show_thumbs(self.plan)

    def run(self):
        if not self.plan:
            return
        s = self.plan.settings
        if not messagebox.askyesno(
                '確認',
                f'{self.plan.n_picked:,} 枚を {self.plan.out_dir} に書き出します。\n'
                + (f'採用しなかった {self.plan.n_rest:,} 枚を _rest/ に残します。\n'
                   if self.plan.n_rest else '')
                + f'\nファイル名: {s.cam}_YYMMDD_HHMMSS_NNNNN_{s.mod_code}{s.ext}\n\n'
                '実行しますか？'):
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
