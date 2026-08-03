"""detection_dev_ui のタブに描くための入口。

このリポジトリの GUI（dataset_manager_gui.py / dsm/gui/）は Tkinter で、
画面を持たない環境では動かない。ここでは画面だけを Streamlit で書き直し、
処理は dsm/（ops.py・dataset.py）をそのまま呼ぶ。

detection_dev_ui のコードは参照しない。使うのは streamlit だけ。
単体での使い方（CLI・Tkinter GUI）は今までどおり。
"""
from __future__ import annotations

from pathlib import Path

import streamlit as st

# detection_dev_ui でのデータの置き場。環境変数が無ければ既定値を使う。
# （このファイルが単体で読まれても壊れないように、import は関数の中で行う）
import os

DATA_DIR = Path(os.getenv("DATA_DIR", "/workspace/data"))

_KEY = "adt"          # ウィジェットの key が他のタブとぶつからないようにする接頭辞


# ---------------------------------------------------------------------------
# 共通
# ---------------------------------------------------------------------------
def _dataset_choices() -> list[Path]:
    """data/ 直下のディレクトリを候補として並べる"""
    if not DATA_DIR.exists():
        return []
    return sorted(p for p in DATA_DIR.iterdir() if p.is_dir())


def _pick_dataset(key: str) -> Path | None:
    """データセットを選ばせる。一覧に無いものは直接入力もできる。"""
    choices = _dataset_choices()
    labels = [p.name for p in choices] + ["（パスを直接入力）"]
    sel = st.selectbox("対象データセット", labels, key=f"{_KEY}_{key}_sel")

    if sel == "（パスを直接入力）":
        raw = st.text_input("パス", value=str(DATA_DIR), key=f"{_KEY}_{key}_path")
        return Path(raw) if raw.strip() else None
    return DATA_DIR / sel


@st.cache_data(show_spinner=False)
def _load(root_str: str, mtime: float):
    """データセットを読む。mtime を引数に入れて、変わったら読み直させる。"""
    from dsm import dataset as ds_mod

    ds = ds_mod.load(Path(root_str), read_labels=True)
    # Dataset は dataclass なのでそのまま返せるが、
    # キャッシュに載せる都合で必要なものだけ取り出す
    return ds


def _load_with_spinner(root: Path):
    try:
        with st.spinner(f"{root.name} を読み込んでいます…"):
            return _load(str(root), root.stat().st_mtime), ""
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def _show_warnings(ds) -> None:
    for w in getattr(ds, "warnings", []) or []:
        st.warning(f"⚠ {w}")


# ---------------------------------------------------------------------------
# 内訳を見る
# ---------------------------------------------------------------------------
def render_overview() -> None:
    from dsm.dataset import GROUP_KEYS, summary_rows

    st.markdown("#### 📊 データセットの内訳")
    st.caption(
        "クラス・カメラ・取得日・セッション別に枚数を数えます。"
        "偏りがあると学習が引っ張られるので、分割や追加撮影の判断材料にしてください。"
    )

    root = _pick_dataset("ov")
    if root is None:
        return
    if not root.exists():
        st.error(f"ありません: `{root}`")
        return

    ds, err = _load_with_spinner(root)
    if ds is None:
        st.error(f"読み込めませんでした: {err}")
        return
    _show_warnings(ds)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("画像", ds.n_images)
    c2.metric("ラベルあり", ds.n_labeled)
    c3.metric("ラベルなし", ds.n_images - ds.n_labeled)
    c4.metric("クラス数", len(ds.classes))
    st.caption(f"構造: `{ds.layout or '不明'}`　"
               f"クラス: {', '.join(ds.classes) if ds.classes else '（定義なし）'}")

    if not ds.items:
        st.info("画像が見つかりませんでした。")
        return

    keys = list(GROUP_KEYS)
    key = st.selectbox(
        "何で集計するか", keys,
        format_func=lambda k: GROUP_KEYS[k],
        key=f"{_KEY}_ov_key",
    )

    rows = summary_rows(ds.items, key, ds.classes)
    if not rows:
        st.info("集計できる値がありませんでした。")
        return

    import pandas as pd

    df = pd.DataFrame(rows, columns=[GROUP_KEYS[key], "枚数", "割合(%)"])
    df["割合(%)"] = df["割合(%)"].round(1)
    st.dataframe(df, use_container_width=True, hide_index=True)

    if len(df) > 1:
        st.bar_chart(df.set_index(GROUP_KEYS[key])["枚数"])
        top, bottom = df.iloc[0], df.iloc[-1]
        if bottom["枚数"] and top["枚数"] / bottom["枚数"] >= 5:
            st.warning(
                f"⚠ 偏りがあります: 「{top[GROUP_KEYS[key]]}」{top['枚数']}枚 に対して "
                f"「{bottom[GROUP_KEYS[key]]}」は {bottom['枚数']}枚 です。"
            )


# ---------------------------------------------------------------------------
# 分割する
# ---------------------------------------------------------------------------
def render_split() -> None:
    from dsm import ops
    from dsm.ops import GROUP_MODES, LAYOUTS

    st.markdown("#### ✂️ train / val / test に分ける")
    st.caption(
        "元のデータセットには手を加えず、新しいディレクトリに書き出します。"
    )

    root = _pick_dataset("sp")
    if root is None:
        return
    if not root.exists():
        st.error(f"ありません: `{root}`")
        return

    ds, err = _load_with_spinner(root)
    if ds is None:
        st.error(f"読み込めませんでした: {err}")
        return
    _show_warnings(ds)
    if not ds.items:
        st.info("画像が見つかりませんでした。")
        return

    st.caption(f"対象 {len(ds.items)} 枚（ラベルあり {ds.n_labeled} 枚）")

    # --- 分け方 ---
    mode = st.radio(
        "決め方", ["ratio", "count"],
        format_func=lambda m: "割合で決める" if m == "ratio" else "枚数で決める",
        horizontal=True, key=f"{_KEY}_sp_mode",
    )

    if mode == "ratio":
        c1, c2, c3 = st.columns(3)
        r_tr = c1.number_input("train", 0.0, 1.0, 0.8, 0.05, key=f"{_KEY}_sp_rtr")
        r_va = c2.number_input("val", 0.0, 1.0, 0.2, 0.05, key=f"{_KEY}_sp_rva")
        r_te = c3.number_input("test", 0.0, 1.0, 0.0, 0.05, key=f"{_KEY}_sp_rte")
        total = round(r_tr + r_va + r_te, 4)
        if abs(total - 1.0) > 1e-6:
            st.error(f"合計が 1.0 になっていません（いま {total}）")
            return
        ratios, counts = (r_tr, r_va, r_te), (0, 0, 0)
    else:
        c1, c2, c3 = st.columns(3)
        n_tr = c1.number_input("train", 0, 1000000, max(len(ds.items) - 100, 0),
                               10, key=f"{_KEY}_sp_ctr")
        n_va = c2.number_input("val", 0, 1000000, min(100, len(ds.items)),
                               10, key=f"{_KEY}_sp_cva")
        n_te = c3.number_input("test", 0, 1000000, 0, 10, key=f"{_KEY}_sp_cte")
        ratios, counts = (0.8, 0.2, 0.0), (n_tr, n_va, n_te)

    # --- まとまりの単位 ---
    group = st.radio(
        "ひとまとまりとして扱う単位", list(GROUP_MODES),
        format_func=lambda g: GROUP_MODES[g],
        key=f"{_KEY}_sp_group",
    )
    if group == "none":
        st.caption(
            "ℹ 連写や同一場面の画像が train と val の両方に入ると、"
            "評価が実力より高く出ます。心当たりがあれば「セッション」を選んでください。"
        )

    c1, c2 = st.columns(2)
    with c1:
        stratified = st.checkbox(
            "クラスの比率を保つ（stratified）", value=False, key=f"{_KEY}_sp_strat",
            help="少ないクラスが片方に寄るのを防ぎます")
        seed = st.number_input("乱数シード", 0, 99999, 42, key=f"{_KEY}_sp_seed",
                               help="同じ値なら同じ分け方になります")
    with c2:
        layout = st.selectbox(
            "出力の形", list(LAYOUTS), index=list(LAYOUTS).index("yolo"),
            format_func=lambda l: LAYOUTS[l], key=f"{_KEY}_sp_layout")
        prefix = st.checkbox(
            "ファイル名に取得元を付ける", value=False, key=f"{_KEY}_sp_prefix",
            help="複数のデータセットを混ぜたときの取り違えを防ぎます")

    out_name = st.text_input(
        "出力先（data/ 配下の名前）", value=f"{root.name}_split",
        key=f"{_KEY}_sp_out")
    out_dir = DATA_DIR / out_name.strip() if out_name.strip() else None
    if out_dir is None:
        st.warning("出力先の名前を入れてください。")
        return
    st.caption(f"書き出し先: `{out_dir}`")
    if out_dir.exists() and any(out_dir.iterdir()):
        st.error("その名前は既に使われています。別の名前にしてください。")
        return

    # --- 下見 ---
    if st.button("👁 分け方を確認する", key=f"{_KEY}_sp_preview",
                 use_container_width=True):
        try:
            split_map = ops.split_items(
                ds.items, mode=mode, ratios=ratios, counts=counts,
                stratified=stratified, group=group, seed=int(seed))
            plan = ops.plan_output(split_map, out_dir, layout=layout,
                                   prefix_source=prefix)
        except Exception as e:
            st.error(f"計画を作れませんでした: {type(e).__name__}: {e}")
            return
        st.session_state[f"{_KEY}_sp_plan"] = {
            "counts": dict(plan.counts_by_split()),
            "n": plan.n,
            "renamed": plan.renamed,
            "skipped": [(getattr(i, "stem", "?"), why) for i, why in plan.skipped],
        }

    prev = st.session_state.get(f"{_KEY}_sp_plan")
    if prev:
        st.markdown("**この内容で書き出します**")
        st.markdown("　".join(f"`{k}` {v}枚" for k, v in prev["counts"].items())
                    or "（対象なし）")
        if prev["renamed"]:
            st.caption(f"名前の重複により {prev['renamed']} 件をリネームします")
        if prev["skipped"]:
            st.warning(f"⚠ {len(prev['skipped'])} 件を飛ばします")
            with st.expander("飛ばす対象"):
                for stem, why in prev["skipped"][:50]:
                    st.caption(f"・{stem} — {why}")

        if st.button(f"✂️ {prev['n']} 枚を書き出す", type="primary",
                     use_container_width=True, key=f"{_KEY}_sp_run"):
            bar = st.progress(0.0, text="書き出しています…")

            def _on_progress(done, total):
                if total:
                    bar.progress(min(done / total, 1.0),
                                 text=f"{done} / {total} 件")

            try:
                split_map = ops.split_items(
                    ds.items, mode=mode, ratios=ratios, counts=counts,
                    stratified=stratified, group=group, seed=int(seed))
                plan = ops.plan_output(split_map, out_dir, layout=layout,
                                       prefix_source=prefix)
                stats = ops.write_plan(plan, classes=ds.classes, move=False,
                                       on_progress=_on_progress)
                ops.write_class_files(plan, ds.classes)
            except Exception as e:
                bar.empty()
                st.error(f"書き出しに失敗しました: {type(e).__name__}: {e}")
                return
            bar.empty()

            st.success(
                f"✅ 画像 {stats['images']} 枚 / ラベル {stats['labels']} 件を "
                f"`{out_dir.name}` に書き出しました"
            )
            if stats["errors"]:
                st.warning(f"⚠ {len(stats['errors'])} 件で問題が起きました")
                with st.expander("内容"):
                    for src, msg in stats["errors"][:50]:
                        st.caption(f"・{src} — {msg}")
            st.caption(
                "「📁 データ管理」に出てきます。状態は 🟡 作成中 になっているので、"
                "確認できたら更新しておくと後で分かりやすくなります。"
            )
            st.session_state.pop(f"{_KEY}_sp_plan", None)


# 既定の入口。マニフェストで function を省いたときはこれが呼ばれる。
def render() -> None:
    render_overview()
