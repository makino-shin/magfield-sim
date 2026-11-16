"""
磁界ビューア（Streamlit 版）

準備:
    pip install -r requirements.txt

起動:
    streamlit run magnet_app.py
    → ブラウザが自動で開きます（開かなければ http://localhost:8501 ）

単位: 長さ mm / 磁場 mT / 残留磁束密度 T
"""
import json
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.spatial.transform import Rotation
import magpylib as mp

st.set_page_config(page_title="磁界ビューア", layout="wide")

# 日本語フォント（あるものを使う）
from matplotlib import font_manager
try:
    import matplotlib_fontja  # noqa: F401  クラウド（日本語フォントのないLinux）用。pip で入るフォント
except ImportError:
    pass
_jp = [f for f in ["Yu Gothic", "Meiryo", "Hiragino Sans", "Noto Sans CJK JP", "IPAexGothic", "IPAGothic"]
       if f in {x.name for x in font_manager.fontManager.ttflist}]
plt.rcParams["font.family"] = _jp[:1] + ["sans-serif"]

SHAPES = ["角形", "円柱"]
DIRS = ["+X", "-X", "+Y", "-Y", "+Z", "-Z"]
DIR_VEC = {"+X": (1, 0, 0), "-X": (-1, 0, 0), "+Y": (0, 1, 0),
           "-Y": (0, -1, 0), "+Z": (0, 0, 1), "-Z": (0, 0, -1)}

COLUMNS = ["表示", "名前", "形状", "寸法1", "寸法2", "寸法3", "Br[T]", "着磁方向",
           "X", "Y", "Z", "回転X", "回転Y", "回転Z"]

# 表の「元データ」（起動時の初期値）。編集のたびに書き換えてはいけない
# （書き換えると data_editor の編集状態がリセットされ、1回目の編集が打ち消される）。
if "magnets" not in st.session_state:
    st.session_state.magnets = pd.DataFrame([
        [True, "磁石1", "角形", 5.0, 6.0, 10.0, 1.2, "-X", 10.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ], columns=COLUMNS)
SENSOR_COLUMNS = ["表示", "名前", "X", "Y", "Z"]
if "sensors" not in st.session_state:   # センサの表の元データ（磁石と同じく、編集結果は書き戻さない）
    st.session_state.sensors = pd.DataFrame([[True, "センサ1", 0.0, 0.0, 0.0]], columns=SENSOR_COLUMNS)

# 表示設定の初期値（ウィジェットの key と同じ名前で session_state に置く）
PLANES = ["XZ（Y=一定）", "XY（Z=一定）", "YZ（X=一定）"]
VIEW_DEFAULTS = {"plane": PLANES[0], "offset": 0.0, "half": 25, "res": 160, "density": 1.5, "show_bg": True}
for k, v in VIEW_DEFAULTS.items():
    st.session_state.setdefault(k, v)
st.session_state.setdefault("ver", 0)   # 設定ファイルを読み込むたびに増やし、表をリセットする

# ---------- 設定ファイル（保存・読み込み） ----------
MAGNET_DEFAULTS = {"表示": True, "名前": "磁石", "形状": "角形", "寸法1": 5.0, "寸法2": 5.0, "寸法3": 5.0,
                   "Br[T]": 1.2, "着磁方向": "+Z", "X": 0.0, "Y": 0.0, "Z": 0.0,
                   "回転X": 0.0, "回転Y": 0.0, "回転Z": 0.0}
SENSOR_DEFAULTS = {"表示": True, "名前": "センサ", "X": 0.0, "Y": 0.0, "Z": 0.0}

def records(df):
    """DataFrame → JSON にできる素の Python 型のリスト"""
    out = []
    for r in df.to_dict("records"):
        row = {}
        for k, v in r.items():
            if hasattr(v, "item"):
                v = v.item()          # numpy の数値・bool → Python の型
            if isinstance(v, float) and np.isnan(v):
                v = None
            row[k] = v
        out.append(row)
    return out

def make_config(magnets_df, sensors_df):
    return {
        "app": "magnet_viewer", "version": 1,
        "view": {k: st.session_state[k] for k in VIEW_DEFAULTS},
        "magnets": records(magnets_df),
        "sensors": records(sensors_df),
    }

def table_from(rows, columns, defaults):
    rows = [{c: (r.get(c) if r.get(c) is not None else defaults[c]) for c in columns} for r in rows or []]
    return pd.DataFrame(rows, columns=columns)

def apply_config(cfg):
    if not isinstance(cfg, dict) or "magnets" not in cfg:
        raise ValueError("磁界ビューアの設定ファイルではないようです")
    st.session_state.magnets = table_from(cfg.get("magnets"), COLUMNS, MAGNET_DEFAULTS)
    st.session_state.sensors = table_from(cfg.get("sensors"), SENSOR_COLUMNS, SENSOR_DEFAULTS)
    view = cfg.get("view", {})
    for k, d in VIEW_DEFAULTS.items():
        v = view.get(k, d)
        if k == "plane" and v not in PLANES:
            v = d
        st.session_state[k] = type(d)(v)

# ---------- 磁石オブジェクトをつくる ----------
def build_magnets(df):
    """表の各行から magpylib の磁石を作る。(磁石, 行データ) のリストを返す"""
    out = []
    for _, r in df.iterrows():
        if not bool(r.get("表示", True)):
            continue
        try:
            br = float(r["Br[T]"])
            pol = tuple(br * np.array(DIR_VEC.get(r["着磁方向"], (0, 0, 1))))
            rot = Rotation.from_euler("xyz", [float(r["回転X"]), float(r["回転Y"]),
                                              float(r["回転Z"])], degrees=True)
            pos = (float(r["X"]), float(r["Y"]), float(r["Z"]))
            if r["形状"] == "円柱":
                d, h = float(r["寸法1"]), float(r["寸法2"])
                if d <= 0 or h <= 0:
                    continue
                m = mp.magnet.Cylinder(polarization=pol, dimension=(d, h),
                                       position=pos, orientation=rot)
            else:
                a, b, c = float(r["寸法1"]), float(r["寸法2"]), float(r["寸法3"])
                if min(a, b, c) <= 0:
                    continue
                m = mp.magnet.Cuboid(polarization=pol, dimension=(a, b, c),
                                     position=pos, orientation=rot)
            out.append((m, r))
        except (ValueError, TypeError):
            continue  # 入力途中の行などは無視
    return out

def inside_mask(mags, pts):
    """各点がどれかの磁石の内部にあるか（断面の塗りと、内部の磁力線を消すのに使う）"""
    mask = np.zeros(len(pts), dtype=bool)
    for m, _ in mags:
        local = m.orientation.inv().apply(pts - m.position)  # 磁石の座標系に変換
        if isinstance(m, mp.magnet.Cylinder):
            d, h = m.dimension
            mask |= (np.hypot(local[:, 0], local[:, 1]) <= d / 2) & (np.abs(local[:, 2]) <= h / 2)
        else:
            mask |= np.all(np.abs(local) <= np.array(m.dimension) / 2, axis=1)
    return mask

# ---------- 画面: サイドバー ----------
with st.sidebar:
    st.header("設定ファイル")
    save_box = st.container()   # 保存ボタンは表の内容が決まった後（ページ下部の処理）でここに置く
    up = st.file_uploader("設定ファイルを読み込む", type="json", key=f"uploader_{st.session_state.ver}",
                          help="以前に保存した JSON ファイルを選ぶと、磁石・センサ・表示設定がその状態に戻ります")
    if up is not None:
        try:
            apply_config(json.loads(up.getvalue().decode("utf-8")))
            st.session_state.ver += 1          # 表とアップローダーをリセット（同じファイルを何度も適用しないため）
            st.session_state.load_msg = f"「{up.name}」を読み込みました"
            st.rerun()
        except (ValueError, UnicodeDecodeError) as e:
            st.error(f"読み込めませんでした：{e}")
    if "load_msg" in st.session_state:
        st.success(st.session_state.pop("load_msg"))

    st.header("表示の設定")
    plane = st.selectbox("断面", PLANES, key="plane")
    offset = st.number_input("断面の位置 [mm]", step=0.5, key="offset")
    half = st.slider("表示範囲 ±[mm]", 5, 100, key="half")
    res = st.slider("格子の細かさ", 60, 300, step=20, key="res",
                    help="大きいほどきれいだが計算が遅くなる")
    density = st.slider("磁力線の密度", 0.5, 4.0, step=0.1, key="density")
    show_bg = st.checkbox("磁場の強さを背景色で表示", key="show_bg")


# ---------- 画面: 磁石の表 ----------
st.title("磁界ビューア")
st.caption("表の最下行の ＋ で磁石を追加、行を選んで 🗑 で削除。"
           "寸法は 角形=縦×横×奥行(X,Y,Z)、円柱=直径×高さ（寸法3は無視）。")

edited = st.data_editor(
    st.session_state.magnets,
    key=f"magnet_editor_{st.session_state.ver}",
    num_rows="dynamic",
    width="stretch",
    column_config={
        "表示": st.column_config.CheckboxColumn(default=True),
        "名前": st.column_config.TextColumn(default="磁石"),
        "形状": st.column_config.SelectboxColumn(options=SHAPES, default="角形", required=True),
        "寸法1": st.column_config.NumberColumn("寸法1[mm]", default=5.0, min_value=0.0, step=0.5),
        "寸法2": st.column_config.NumberColumn("寸法2[mm]", default=5.0, min_value=0.0, step=0.5),
        "寸法3": st.column_config.NumberColumn("寸法3[mm]", default=5.0, min_value=0.0, step=0.5),
        "Br[T]": st.column_config.NumberColumn(default=1.2, min_value=0.0, max_value=2.0, step=0.05,
                                               help="N35〜N38 ≒ 1.2、フェライト ≒ 0.4"),
        "着磁方向": st.column_config.SelectboxColumn(options=DIRS, default="+Z", required=True,
                                                 help="N極が向く方向（磁石自身の座標で）"),
        "X": st.column_config.NumberColumn("X[mm]", default=0.0, step=0.5),
        "Y": st.column_config.NumberColumn("Y[mm]", default=0.0, step=0.5),
        "Z": st.column_config.NumberColumn("Z[mm]", default=0.0, step=0.5),
        "回転X": st.column_config.NumberColumn("回転X[°]", default=0.0, step=5.0),
        "回転Y": st.column_config.NumberColumn("回転Y[°]", default=0.0, step=5.0),
        "回転Z": st.column_config.NumberColumn("回転Z[°]", default=0.0, step=5.0),
    },
)
mags = build_magnets(edited)   # 編集結果は戻り値 edited をそのまま使う

st.subheader("センサ")
st.caption("磁石と同じく、最下行の ＋ で追加、🗑 で削除。断面から外れたセンサは薄く表示されます。")
sensors = st.data_editor(
    st.session_state.sensors,
    key=f"sensor_editor_{st.session_state.ver}",
    num_rows="dynamic",
    width="stretch",
    column_config={
        "表示": st.column_config.CheckboxColumn(default=True),
        "名前": st.column_config.TextColumn(default="センサ"),
        "X": st.column_config.NumberColumn("X[mm]", default=0.0, step=0.5),
        "Y": st.column_config.NumberColumn("Y[mm]", default=0.0, step=0.5),
        "Z": st.column_config.NumberColumn("Z[mm]", default=0.0, step=0.5),
    },
)
with save_box:
    st.download_button(
        "現在の設定を保存（JSON）",
        data=json.dumps(make_config(edited, sensors), ensure_ascii=False, indent=2),
        file_name=f"magnet_viewer_{datetime.now(ZoneInfo('Asia/Tokyo')):%Y%m%d_%H%M}.json",
        mime="application/json", width="stretch",
    )

sensor_list = []
for i, r in sensors.iterrows():
    try:
        if not bool(r.get("表示", True)):
            continue
        name = str(r["名前"]) if pd.notna(r["名前"]) else f"センサ{i + 1}"
        sensor_list.append((name, np.array([float(r["X"]), float(r["Y"]), float(r["Z"])])))
    except (ValueError, TypeError):
        continue  # 入力途中の行は無視

if not mags:
    st.info("磁石がありません。表の ＋ で追加してください。")
    st.stop()

col = mp.Collection(*[m for m, _ in mags])

# ---------- 計算 ----------
u = np.linspace(-half, half, res)          # 図の横軸
v = np.linspace(-half, half, res)          # 図の縦軸
U, V = np.meshgrid(u, v)
if plane.startswith("XZ"):
    # 横 = X、縦 = Z（Z が上）
    pts = np.stack([U.ravel(), np.full(U.size, offset), V.ravel()], axis=1)
    iu, iv, labels = 0, 2, ("X [mm]", "Z [mm]")
elif plane.startswith("XY"):
    pts = np.stack([U.ravel(), V.ravel(), np.full(U.size, offset)], axis=1)
    iu, iv, labels = 0, 1, ("X [mm]", "Y [mm]")
else:
    pts = np.stack([np.full(U.size, offset), U.ravel(), V.ravel()], axis=1)
    iu, iv, labels = 1, 2, ("Y [mm]", "Z [mm]")

B = col.getB(pts) * 1000                    # T → mT（全点を一括計算）
Bu = B[:, iu].reshape(U.shape)
Bv = B[:, iv].reshape(U.shape)
Bmag = np.linalg.norm(B, axis=1).reshape(U.shape)
inside = inside_mask(mags, pts).reshape(U.shape)

# ---------- 描画 ----------
fig, ax = plt.subplots(figsize=(8, 7))
if show_bg:
    bg = np.ma.masked_where(inside, np.clip(Bmag, 0.1, None))
    pc = ax.pcolormesh(u, v, bg, norm=LogNorm(vmin=0.1, vmax=max(bg.max(), 1)),
                       cmap="Blues", shading="auto", alpha=0.6)
    fig.colorbar(pc, ax=ax, label="|B| [mT]", shrink=0.8)

Bu_m = np.ma.masked_where(inside, Bu)       # 磁石の内部は磁力線を描かない
Bv_m = np.ma.masked_where(inside, Bv)
ax.streamplot(u, v, Bu_m, Bv_m, density=density, color="0.25",
              linewidth=0.7, arrowsize=0.8)

# 磁石の断面（N極側=赤、S極側=青）
for m, r in mags:
    local = m.orientation.inv().apply(pts - m.position)
    if isinstance(m, mp.magnet.Cylinder):
        d, h = m.dimension
        ins = (np.hypot(local[:, 0], local[:, 1]) <= d / 2) & (np.abs(local[:, 2]) <= h / 2)
    else:
        ins = np.all(np.abs(local) <= np.array(m.dimension) / 2, axis=1)
    if not ins.any():
        continue
    mdir = np.array(DIR_VEC.get(r["着磁方向"], (0, 0, 1)))
    side = np.where(local @ mdir >= 0, 1.0, -1.0)   # 着磁方向の前半分=N, 後ろ半分=S
    img = np.where(ins, side, np.nan).reshape(U.shape)
    ax.contourf(u, v, img, levels=[-1.5, 0, 1.5], colors=["#85B7EB", "#F09595"], zorder=3)
    ax.contour(u, v, ins.reshape(U.shape).astype(float), levels=[0.5],
               colors="k", linewidths=0.6, zorder=4)
    ctr = m.position
    ax.text(ctr[iu], ctr[iv], str(r["名前"]), ha="center", va="center",
            fontsize=8, zorder=5)

# センサ（複数）
SENSOR_COLORS = ["#1D9E75", "#D85A30", "#7F77DD", "#BA7517", "#D4537E", "#378ADD"]
readings = []
fixed_axis = 3 - iu - iv   # 断面に垂直な軸
for k, (name, spos) in enumerate(sensor_list):
    c = SENSOR_COLORS[k % len(SENSOR_COLORS)]
    sb = col.getB(spos) * 1000
    on_plane = abs(spos[fixed_axis] - offset) < 1e-6
    a = 1.0 if on_plane else 0.35
    ax.plot(spos[iu], spos[iv], "s", color=c, ms=8, alpha=a, zorder=6)
    inplane = np.array([sb[iu], sb[iv]])
    if np.linalg.norm(inplane) > 1e-9:
        d = inplane / np.linalg.norm(inplane) * half * 0.15
        ax.annotate("", xy=(spos[iu] + d[0], spos[iv] + d[1]), xytext=(spos[iu], spos[iv]),
                    arrowprops=dict(arrowstyle="->", color=c, lw=2.5, alpha=a), zorder=7)
    ax.annotate(name, (spos[iu], spos[iv]), xytext=(6, -12), textcoords="offset points",
                fontsize=8, color=c, alpha=a, zorder=7)
    readings.append({
        "名前": name, "色": c,
        "|B| [mT]": np.linalg.norm(sb), "Bx": sb[0], "By": sb[1], "Bz": sb[2],
        "XY面内の向き [°]": np.degrees(np.arctan2(sb[1], sb[0])),
        "Z方向への傾き [°]": np.degrees(np.arctan2(sb[2], np.hypot(sb[0], sb[1]))),
    })

ax.set_xlim(-half, half)
ax.set_ylim(-half, half)
ax.set_aspect("equal")
ax.set_xlabel(labels[0])
ax.set_ylabel(labels[1])
ax.set_title(f"{plane.split('（')[0]} 断面（{plane.split('（')[1].rstrip('）').replace('一定', f'{offset:g}mm')}）")

st.pyplot(fig)

st.subheader("センサの読み")
if readings:
    df_read = pd.DataFrame(readings)
    st.dataframe(
        df_read.style.format({c: "{:.2f}" for c in df_read.columns if c not in ("名前", "色")})
                     .map(lambda v: f"color: {v}; font-weight: bold", subset=["色"]),
        hide_index=True, width="stretch",
    )
    st.caption("色の列は図のセンサの色。矢印は断面内での磁場の向き。"
               "XY面内の向き＝ひねり角、Z方向への傾き＝前後位置に相当。")
else:
    st.info("センサがありません。センサの表の ＋ で追加してください。")
