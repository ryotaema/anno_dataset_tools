"""名前変更タブ — フォルダ内の画像とラベルの名前を一括で変更する。

セッションが違っても `1color.jpg` のような同じ名前が使われていると、1つの
フォルダに集約した時点で衝突する。ファイル名に「どのデータか」を書き込んでおく。

    260101_lab/color/1color.jpg  →  260101_lab_1color_c.jpg

親フォルダ名からの提案が入るが、そのまま書き換えられる。変更後の名前を
プレビューで確認してから実行し、実行内容は rename_log.json に残るので
「元に戻す」でいつでも戻せる。
"""

import tkinter as tk
from pathlib import Path
from tkinter import ttk, messagebox

from .. import rename as rn
from .common import PathRow

PREVIEW_LIMIT = 500


class RenameTab(ttk.Frame):
    def __init__(self, master, state):
        super().__init__(master)
        self.state = state
        self.plan = None
        self._suggested = {}       # 直前に流し込んだ提案（手で直したか判定するため）

        # --- 対象 ---
        tgt = ttk.LabelFrame(self, text='対象')
        tgt.pack(fill='x', padx=8, pady=(8, 4))
        self.dir_row = PathRow(tgt, 'フォルダ:', on_change=self._on_dir)
        self.dir_row.pack(fill='x', padx=8, pady=(8, 2))

        r1 = ttk.Frame(tgt)
        r1.pack(fill='x', padx=8, pady=(2, 8))
        self.recursive = tk.BooleanVar(value=True)
        ttk.Checkbutton(r1, text='サブフォルダも対象にする', variable=self.recursive,
                        command=self._invalidate).pack(side='left')
        self.with_labels = tk.BooleanVar(value=True)
        ttk.Checkbutton(r1, text='ラベル（.txt / .xml）も同じ名前にする',
                        variable=self.with_labels,
                        command=self._invalidate).pack(side='left', padx=(14, 0))
        self.skip_converted = tk.BooleanVar(value=False)
        ttk.Checkbutton(r1, text='すでに標準命名のファイルは変えない',
                        variable=self.skip_converted,
                        command=self._invalidate).pack(side='left', padx=(14, 0))
        ttk.Button(r1, text='提案を入れ直す', command=self.apply_suggestion).pack(side='right')

        # --- 名前の付け方 ---
        nm = ttk.LabelFrame(self, text='名前の付け方')
        nm.pack(fill='x', padx=8, pady=4)
        self.mode_var = tk.StringVar(value='affix')

        r2 = ttk.Frame(nm)
        r2.pack(fill='x', padx=8, pady=(8, 2))
        ttk.Radiobutton(r2, text='かんたん', value='affix', variable=self.mode_var,
                        command=self._on_mode).pack(side='left')
        ttk.Label(r2, text='先頭:').pack(side='left', padx=(10, 0))
        self.prefix_var = tk.StringVar()
        self.prefix_entry = ttk.Entry(r2, textvariable=self.prefix_var, width=26)
        self.prefix_entry.pack(side='left', padx=(4, 8))
        ttk.Label(r2, text='元の名前').pack(side='left')
        ttk.Label(r2, text='末尾:').pack(side='left', padx=(8, 0))
        self.suffix_var = tk.StringVar()
        self.suffix_entry = ttk.Entry(r2, textvariable=self.suffix_var, width=16)
        self.suffix_entry.pack(side='left', padx=(4, 0))

        r3 = ttk.Frame(nm)
        r3.pack(fill='x', padx=8, pady=2)
        ttk.Radiobutton(r3, text='テンプレート', value='template', variable=self.mode_var,
                        command=self._on_mode).pack(side='left')
        self.template_var = tk.StringVar(value='{parent2}_{stem}_{mod}')
        self.template_entry = ttk.Entry(r3, textvariable=self.template_var, width=46)
        self.template_entry.pack(side='left', padx=(10, 8))
        ttk.Label(r3, text='使えるトークン: ' + ' '.join(t for t, _ in rn.TOKENS),
                  foreground='#555').pack(side='left')

        r4 = ttk.Frame(nm)
        r4.pack(fill='x', padx=8, pady=(2, 4))
        ttk.Radiobutton(r4, text='標準命名', value='standard', variable=self.mode_var,
                        command=self._on_mode).pack(side='left')
        ttk.Label(r4, text='カメラ:').pack(side='left', padx=(10, 0))
        self.cam_var = tk.StringVar(value='d435')
        self.cam_entry = ttk.Combobox(r4, textvariable=self.cam_var, width=8,
                                      values=['d435', 'd405', 'nyx', 'vid'])
        self.cam_entry.pack(side='left', padx=(4, 8))
        ttk.Label(r4, text='日時:').pack(side='left')
        self.dt_var = tk.StringVar(value='260101_090000')
        self.dt_entry = ttk.Entry(r4, textvariable=self.dt_var, width=16)
        self.dt_entry.pack(side='left', padx=(4, 8))
        ttk.Label(r4, text='→ {cam}_{YYMMDD}_{HHMMSS}_{NNNNN}_{mod}'
                          '（セッション単位の分割・カメラ別集計が使えるようになります）',
                  foreground='#555').pack(side='left')

        self.hint_var = tk.StringVar(value='フォルダを選ぶと名前の候補を提案します')
        ttk.Label(nm, textvariable=self.hint_var, foreground='#357').pack(
            anchor='w', padx=12, pady=(0, 6))

        for v in (self.prefix_var, self.suffix_var, self.template_var,
                  self.cam_var, self.dt_var):
            v.trace_add('write', lambda *a: self._invalidate())

        # --- 実行方法 ---
        how = ttk.LabelFrame(self, text='実行方法')
        how.pack(fill='x', padx=8, pady=4)
        r5 = ttk.Frame(how)
        r5.pack(fill='x', padx=8, pady=(8, 2))
        self.apply_mode = tk.StringVar(value='inplace')
        ttk.Radiobutton(r5, text='その場で名前を変更する', value='inplace',
                        variable=self.apply_mode,
                        command=self._on_apply_mode).pack(side='left')
        ttk.Radiobutton(r5, text='別フォルダにコピーする（原本を残す）', value='copy',
                        variable=self.apply_mode,
                        command=self._on_apply_mode).pack(side='left', padx=(14, 0))
        self.copy_row = PathRow(how, '出力先:', on_change=self._invalidate)
        self.copy_row.pack(fill='x', padx=8, pady=2)

        r6 = ttk.Frame(how)
        r6.pack(fill='x', padx=8, pady=(2, 8))
        self.dedupe = tk.BooleanVar(value=True)
        ttk.Checkbutton(r6, text='名前が重なったら連番を付けて回避する', variable=self.dedupe,
                        command=self._invalidate).pack(side='left')
        self.lower_ext = tk.BooleanVar(value=False)
        ttk.Checkbutton(r6, text='拡張子を小文字にそろえる', variable=self.lower_ext,
                        command=self._invalidate).pack(side='left', padx=(14, 0))

        # --- 実行ボタン（先に下端を確保する） ---
        btns = ttk.Frame(self)
        btns.pack(side='bottom', fill='x', padx=8, pady=(4, 8))
        self.run_btn = ttk.Button(btns, text='② 名前を変更', command=self.run)
        self.run_btn.pack(side='left')
        self.run_btn.state(['disabled'])
        ttk.Button(btns, text='元に戻す（最後の変更を取り消す）',
                   command=self.undo).pack(side='left', padx=(8, 0))

        # --- プレビュー ---
        prev = ttk.LabelFrame(self, text='確認')
        prev.pack(fill='both', expand=True, padx=8, pady=4)
        bar = ttk.Frame(prev)
        bar.pack(fill='x', padx=8, pady=(8, 4))
        ttk.Button(bar, text='① プレビューを作成', command=self.preview).pack(side='left')
        self.summary_var = tk.StringVar(value='フォルダを指定してください')
        ttk.Label(bar, textvariable=self.summary_var).pack(side='left', padx=(12, 0))

        body = ttk.Frame(prev)
        body.pack(fill='both', expand=True, padx=8, pady=(0, 8))
        self.tree = ttk.Treeview(body, columns=('after', 'where'), height=10)
        self.tree.heading('#0', text='変更前')
        self.tree.heading('after', text='変更後')
        self.tree.heading('where', text='場所')
        self.tree.column('#0', width=260, stretch=True)
        self.tree.column('after', width=300, stretch=True)
        self.tree.column('where', width=200, stretch=False)
        self.tree.tag_configure('label', foreground='#357')
        self.tree.tag_configure('conflict', foreground='#b00')
        self.tree.tag_configure('same', foreground='#888')
        vsb = ttk.Scrollbar(body, orient='vertical', command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        self.tree.pack(side='left', fill='both', expand=True)
        vsb.pack(side='left', fill='y')

        self._on_mode()
        self._on_apply_mode()

    # ------------------------------------------------------------------ 提案

    def _on_dir(self, *_):
        self._invalidate()
        d = self.dir_row.get()
        if not d or not Path(d).expanduser().is_dir():
            return
        # 手で直していないときだけ提案を入れ直す
        untouched = (self.prefix_var.get() == self._suggested.get('prefix', '')
                     and self.suffix_var.get() == self._suggested.get('suffix', '')
                     and self.template_var.get() == self._suggested.get(
                         'template', '{parent2}_{stem}_{mod}'))
        if untouched or not self._suggested:
            self.apply_suggestion()

    def apply_suggestion(self):
        d = self.dir_row.get()
        if not d or not Path(d).expanduser().is_dir():
            messagebox.showinfo('フォルダが未指定', 'フォルダを指定してください。')
            return
        hint = rn.suggest(d)
        self.prefix_var.set(hint['prefix'])
        self.suffix_var.set(hint['suffix'])
        self.template_var.set(hint['template'])
        self._suggested = dict(hint)
        why = f'提案: {hint["why"]}　— そのまま書き換えられます'
        if '{' in hint['prefix'] + hint['suffix']:
            why += '（{mod} はファイルごとに c / d などに置き換わります）'
        self.hint_var.set(why)
        self._invalidate()

    # ------------------------------------------------------------------ 状態

    def _on_mode(self):
        mode = self.mode_var.get()
        for w in (self.prefix_entry, self.suffix_entry):
            w.state(['!disabled'] if mode == 'affix' else ['disabled'])
        self.template_entry.state(['!disabled'] if mode == 'template' else ['disabled'])
        for w in (self.cam_entry, self.dt_entry):
            w.state(['!disabled'] if mode == 'standard' else ['disabled'])
        self._invalidate()

    def _on_apply_mode(self):
        state = 'normal' if self.apply_mode.get() == 'copy' else 'disabled'
        for child in self.copy_row.winfo_children():
            try:
                child.configure(state=state)
            except tk.TclError:
                pass
        self._invalidate()

    def _invalidate(self, *_):
        self.plan = None
        self.run_btn.state(['disabled'])

    def refresh(self):
        pass          # 読み込み済みデータセットには依存しないタブ

    def spec(self):
        mode = self.mode_var.get()
        if mode == 'standard':
            date, _, time = self.dt_var.get().strip().partition('_')
            if not (self.cam_var.get().strip() and len(date) == 6 and len(time) == 6):
                raise ValueError('標準命名にはカメラと日時（YYMMDD_HHMMSS）が必要です')
            return rn.Spec(mode='standard', cam=self.cam_var.get().strip(),
                           date=date, time=time, lower=self.lower_ext.get(),
                           dedupe=self.dedupe.get())
        if mode == 'template':
            if '{stem}' not in self.template_var.get() and \
                    '{n' not in self.template_var.get():
                raise ValueError('テンプレートには {stem} か {n} を入れてください'
                                 '（すべて同じ名前になってしまいます）')
            return rn.Spec(mode='template', template=self.template_var.get().strip(),
                           lower=self.lower_ext.get(), dedupe=self.dedupe.get())
        return rn.Spec(mode='affix', prefix=self.prefix_var.get(),
                       suffix=self.suffix_var.get(), lower=self.lower_ext.get(),
                       dedupe=self.dedupe.get())

    # ------------------------------------------------------------------ 操作

    def preview(self):
        d = self.dir_row.get()
        if not d or not Path(d).expanduser().is_dir():
            messagebox.showinfo('フォルダが未指定', '対象のフォルダを指定してください。')
            return
        copy_mode = self.apply_mode.get() == 'copy'
        if copy_mode and not self.copy_row.get():
            messagebox.showinfo('出力先が未指定', 'コピー先のフォルダを指定してください。')
            return
        try:
            spec = self.spec()
        except ValueError as e:
            messagebox.showerror('設定エラー', str(e))
            return

        self.state.status.message('対象を確認中')
        self.plan = rn.build_plan(
            d, spec, recursive=self.recursive.get(),
            include_labels=self.with_labels.get(),
            skip_converted=self.skip_converted.get(),
            mode='copy' if copy_mode else 'inplace',
            out_dir=self.copy_row.get() if copy_mode else None)
        self.state.status.reset()

        plan = self.plan
        conflict_srcs = {s for _, sources in plan.conflicts for s in sources}

        self.tree.delete(*self.tree.get_children())
        for src, dst, kind in plan.pairs[:PREVIEW_LIMIT]:
            tags = ('conflict',) if src in conflict_srcs else \
                   (('label',) if kind == 'ラベル' else ())
            where = str(dst.parent) if copy_mode else dst.parent.name
            self.tree.insert('', 'end', text=f'[{kind}] {src.name}',
                             values=(dst.name, where), tags=tags)
        if plan.n > PREVIEW_LIMIT:
            self.tree.insert('', 'end', text=f'... 他 {plan.n - PREVIEW_LIMIT:,} 件',
                             values=('', ''), tags=('same',))

        for note in plan.notes:
            self.state.log.write(f'  {note}')

        msg = (f'{plan.n:,} 件を変更（画像 {plan.n_images:,} / ラベル {plan.n_labels:,}）'
               f'　変更なし {plan.unchanged:,} 件')
        if plan.conflicts:
            msg += f'　⚠ 名前の重複 {len(plan.conflicts)} 件 — 実行できません'
            for dst, sources in plan.conflicts[:10]:
                self.state.log.write(
                    f'  ⚠ 重複: {dst.name} ← ' + ', '.join(s.name for s in sources))
        elif plan.n:
            msg += '　▶ 下の「② 名前を変更」で実行します'
        self.summary_var.set(msg)
        self.run_btn.state(['!disabled'] if plan.ok else ['disabled'])

    def run(self):
        if not self.plan or not self.plan.ok:
            return
        plan = self.plan
        copy_mode = plan.mode == 'copy'
        if not messagebox.askyesno(
                '確認',
                f'{plan.n:,} 件の名前を変更します。\n'
                f'（画像 {plan.n_images:,} / ラベル {plan.n_labels:,}）\n\n'
                + (f'コピー先: {plan.out_dir}\n原本はそのまま残ります。\n\n' if copy_mode
                   else f'対象: {plan.root}\nその場で名前を変更します'
                        '（rename_log.json から元に戻せます）。\n\n')
                + '実行しますか？'):
            return

        self.run_btn.state(['disabled'])
        log, status = self.state.log, self.state.status
        log.write(f'名前変更を開始 → {plan.out_dir if copy_mode else plan.root}')

        stats = rn.apply_plan(plan, on_progress=lambda d, t: status.step(d, t, '変更中 '))
        status.reset()

        log.write(f'  {stats["renamed"]:,} 件を'
                  + ('コピーしました' if copy_mode else '変更しました'))
        for src, err in stats['errors'][:10]:
            log.write(f'  [エラー] {src}: {err}')
        if stats['log']:
            log.write(f'  履歴: {stats["log"]}（「元に戻す」で戻せます）')

        out = plan.out_dir if copy_mode else plan.root
        n = stats['renamed']
        self._invalidate()
        self.summary_var.set(f'{n:,} 件を変更しました')
        if messagebox.askyesno('完了',
                               f'{n:,} 件の名前を変更しました。\n\n'
                               'このフォルダをデータセットとして読み込みますか？'):
            ds = self.state.add(out)
            if ds:
                log.write(f'読み込み: {ds.root}  画像={ds.n_images:,}')
                self.state.notify()

    def undo(self):
        d = self.copy_row.get() if self.apply_mode.get() == 'copy' else self.dir_row.get()
        if not d:
            messagebox.showinfo('フォルダが未指定', '対象のフォルダを指定してください。')
            return
        data, path = rn.read_log(d)
        if not data or not data.get('runs'):
            messagebox.showinfo('履歴がありません',
                                f'{Path(d) / rn.LOG_NAME} が見つかりません。\n'
                                'このツールで変更したフォルダを指定してください。')
            return
        last = data['runs'][-1]
        if not messagebox.askyesno(
                '確認',
                f'最後の変更（{last.get("at", "")}／{last.get("renamed", 0):,} 件）を'
                '取り消して、元の名前に戻します。\n\n実行しますか？'):
            return

        status = self.state.status
        n, msg = rn.undo_last(d, on_progress=lambda a, b: status.step(a, b, '復元中 '))
        status.reset()
        self.state.log.write(msg)
        self._invalidate()
        self.summary_var.set(msg)
        messagebox.showinfo('取り消し', msg)
