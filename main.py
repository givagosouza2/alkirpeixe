import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from scipy.io import loadmat

st.set_page_config(page_title='Análise Claro–Escuro', page_icon='🐟', layout='wide')
st.title('Análise de Trajetória 2D — Campo Claro/Escuro')
st.caption('CSV ou .mat. O primeiro X é redefinido como zero; X>0 = escuro e X<0 = claro.')


def read_csv_flexible(uploaded_file):
    raw = uploaded_file.getvalue()
    text = None
    for enc in ('utf-8-sig','utf-8','latin-1'):
        try:
            text = raw.decode(enc); break
        except UnicodeDecodeError:
            pass
    if text is None:
        raise ValueError('Não foi possível decodificar o CSV.')
    try:
        df = pd.read_csv(io.StringIO(text), sep=None, engine='python')
    except Exception:
        df = None
        for sep in (',',';','\t'):
            try:
                c = pd.read_csv(io.StringIO(text), sep=sep)
                if c.shape[1] >= 3:
                    df = c; break
            except Exception:
                pass
    if df is None or df.shape[1] < 3:
        raise ValueError('O CSV deve ter pelo menos 3 colunas: tempo, X e Y.')
    return df


def to_numeric_flexible(series):
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors='coerce')
    s = series.astype(str).str.strip()
    if s.str.contains(',', regex=False).mean() > 0.3 and s.str.contains('.', regex=False).mean() < 0.3:
        s = s.str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
    return pd.to_numeric(s, errors='coerce')


def extract_from_csv(uploaded_file):
    df = read_csv_flexible(uploaded_file)
    return pd.DataFrame({
        'tempo': to_numeric_flexible(df.iloc[:,0]),
        'x': to_numeric_flexible(df.iloc[:,1]),
        'y': to_numeric_flexible(df.iloc[:,2]),
    })


def extract_from_mat(uploaded_file):
    try:
        mat = loadmat(
            io.BytesIO(uploaded_file.getvalue()),
            squeeze_me=True,
            struct_as_record=False
        )
    except NotImplementedError:
        raise ValueError(
            "Arquivo MATLAB v7.3/HDF5 não suportado por scipy.io.loadmat nesta versão."
        )
    except Exception as exc:
        raise ValueError(f"Não foi possível abrir o .mat: {exc}")

    regions = []
    processed_region = None
    stopped_events = pd.DataFrame(columns=['ti', 'tf'])

    if "e" in mat:
        e = mat["e"]

        try:
            df = pd.DataFrame({
                "tempo": np.asarray(e.t, dtype=float).ravel(),
                "x": np.asarray(e.posicao.x, dtype=float).ravel(),
                "y": np.asarray(e.posicao.y, dtype=float).ravel(),
            })
        except Exception as exc:
            raise ValueError(
                "Encontrei 'e', mas não consegui extrair e.t, e.posicao.x e e.posicao.y."
            ) from exc

        # Conversão dos limites de pixels para o mesmo sistema em cm de posicao.x/y.
        # x_cm = x_pixel / px_per_cm_x
        # y_cm = (altura_imagem - y_pixel) / px_per_cm_y
        try:
            px_per_cm_x = float(e.pxcm.x)
            px_per_cm_y = float(e.pxcm.y)
            image_height = float(e.figdimensions.l)

            for i, reg in enumerate(np.atleast_1d(e.areaint), start=1):
                rx_px = np.asarray(reg.x, dtype=float).ravel()
                ry_px = np.asarray(reg.y, dtype=float).ravel()

                rx_cm = rx_px / px_per_cm_x
                ry_cm = (image_height - ry_px) / px_per_cm_y

                if len(rx_cm) >= 3 and len(rx_cm) == len(ry_cm):
                    regions.append({
                        "name": f"Região {i}",
                        "x": rx_cm,
                        "y": ry_cm,
                    })

            try:
                ap = e.areaproc
                ax = np.asarray(ap.x, dtype=float).ravel() / px_per_cm_x
                ay = (
                    image_height - np.asarray(ap.y, dtype=float).ravel()
                ) / px_per_cm_y
                processed_region = {"x": ax, "y": ay}
            except Exception:
                processed_region = None

        except Exception:
            regions = []
            processed_region = None

        # Episódios de imobilidade/parado já detectados no arquivo MATLAB
        try:
            ti = np.atleast_1d(np.asarray(e.parado.ti, dtype=float)).ravel()
            tf = np.atleast_1d(np.asarray(e.parado.tf, dtype=float)).ravel()

            n = min(len(ti), len(tf))
            stopped_events = pd.DataFrame({
                "ti": ti[:n],
                "tf": tf[:n],
            })
            stopped_events = stopped_events[
                np.isfinite(stopped_events["ti"]) &
                np.isfinite(stopped_events["tf"]) &
                (stopped_events["tf"] >= stopped_events["ti"])
            ].reset_index(drop=True)
        except Exception:
            stopped_events = pd.DataFrame(columns=["ti", "tf"])

        return df, regions, processed_region, stopped_events

    keys = {k.lower(): k for k in mat.keys() if not k.startswith("__")}
    tkey = next((keys[k] for k in keys if k in ("t","tempo","time")), None)
    xkey = next((keys[k] for k in keys if k in ("x","posicao_x","position_x")), None)
    ykey = next((keys[k] for k in keys if k in ("y","posicao_y","position_y")), None)

    if tkey and xkey and ykey:
        df = pd.DataFrame({
            "tempo": np.asarray(mat[tkey], dtype=float).ravel(),
            "x": np.asarray(mat[xkey], dtype=float).ravel(),
            "y": np.asarray(mat[ykey], dtype=float).ravel(),
        })
        return df, [], None, pd.DataFrame(columns=['ti', 'tf'])

    raise ValueError("Não encontrei e.t, e.posicao.x e e.posicao.y no .mat.")

def prepare_data(df):
    out = df.copy()
    for c in ('tempo','x','y'):
        out[c] = pd.to_numeric(out[c], errors='coerce')
    out = out.replace([np.inf,-np.inf], np.nan).dropna()
    out = out.sort_values('tempo').drop_duplicates(subset='tempo', keep='first').reset_index(drop=True)
    if len(out) < 2:
        raise ValueError('São necessárias pelo menos duas amostras válidas.')
    out['tempo_original'] = out['tempo']
    out['tempo'] = out['tempo'] - out['tempo'].iloc[0]
    x0 = float(out['x'].iloc[0])
    out['x_original'] = out['x']
    out['x'] = out['x'] - x0
    return out, x0



def point_in_polygon(x, y, poly_x, poly_y):
    """Ray casting para testar se um ponto está dentro de um polígono."""
    px = np.asarray(poly_x, dtype=float)
    py = np.asarray(poly_y, dtype=float)

    inside = False
    j = len(px) - 1

    for i in range(len(px)):
        xi, yi = px[i], py[i]
        xj, yj = px[j], py[j]

        intersects = (
            ((yi > y) != (yj > y))
            and (
                x
                < (xj - xi) * (y - yi) / ((yj - yi) + 1e-15) + xi
            )
        )

        if intersects:
            inside = not inside

        j = i

    return inside


def classify_region(x, y, regions):
    for reg in regions:
        if point_in_polygon(x, y, reg["x"], reg["y"]):
            return reg["name"]
    return "Fora"


def segment_intersection_fraction(p0, p1, q0, q1):
    """Fração u do segmento p0->p1 que intersecta q0->q1."""
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    q0 = np.asarray(q0, dtype=float)
    q1 = np.asarray(q1, dtype=float)

    r = p1 - p0
    s = q1 - q0

    den = r[0] * s[1] - r[1] * s[0]

    if abs(den) < 1e-12:
        return None

    qp = q0 - p0
    u = (qp[0] * s[1] - qp[1] * s[0]) / den
    v = (qp[0] * r[1] - qp[1] * r[0]) / den

    if 0 <= u <= 1 and 0 <= v <= 1:
        return float(u)

    return None


def crossing_fraction_between_regions(p0, p1, regions):
    """Encontra a interseção do vetor com as bordas dos polígonos."""
    candidates = []

    for reg in regions:
        poly = np.column_stack([
            np.asarray(reg["x"], dtype=float),
            np.asarray(reg["y"], dtype=float)
        ])

        if len(poly) < 3:
            continue

        if not np.allclose(poly[0], poly[-1]):
            poly = np.vstack([poly, poly[0]])

        for k in range(len(poly) - 1):
            u = segment_intersection_fraction(
                p0, p1, poly[k], poly[k + 1]
            )
            if u is not None and 1e-8 < u < 1 - 1e-8:
                candidates.append(u)

    if not candidates:
        return None

    # Quando duas regiões compartilham a mesma fronteira pode haver
    # interseções duplicadas. A mediana é robusta nesse caso.
    return float(np.median(candidates))


def split_segments_regions(df, regions, region_to_side):
    """
    Classifica os vetores usando os polígonos reais do .mat.
    Vetores que mudam de região são divididos na fronteira geométrica.
    """
    t = df["tempo"].to_numpy(float)
    xo = df["x_original"].to_numpy(float)
    xn = df["x"].to_numpy(float)
    y = df["y"].to_numpy(float)

    labels = [
        classify_region(xo[i], y[i], regions)
        for i in range(len(df))
    ]

    rows = []
    events = []

    for i in range(len(df) - 1):
        t0, t1 = t[i], t[i + 1]
        dt = t1 - t0
        if dt <= 0:
            continue

        # Geometria original para localizar a fronteira
        p0 = np.array([xo[i], y[i]], dtype=float)
        p1 = np.array([xo[i + 1], y[i + 1]], dtype=float)

        # Coordenada X normalizada é usada nas análises e figuras
        x0, x1 = xn[i], xn[i + 1]
        y0, y1 = y[i], y[i + 1]

        lab0 = labels[i]
        lab1 = labels[i + 1]

        side0 = region_to_side.get(lab0, "Fora")
        side1 = region_to_side.get(lab1, "Fora")

        dx = x1 - x0
        dy = y1 - y0
        full_len = float(np.hypot(dx, dy))
        full_speed = full_len / dt

        if lab0 == lab1:
            angle = float((np.degrees(np.arctan2(dy, dx)) + 360) % 360)

            rows.append([
                side0, t0, t1, dt,
                x0, y0, x1, y1,
                dx, dy, full_len, angle, False
            ])
            continue

        u = crossing_fraction_between_regions(p0, p1, regions)

        if u is None:
            # Se por precisão numérica não for encontrada interseção,
            # usa o ponto médio somente como fallback.
            u = 0.5

        tc = t0 + u * dt
        xc = x0 + u * dx
        yc = y0 + u * dy

        # Primeira parte
        dx1 = xc - x0
        dy1 = yc - y0
        L1 = float(np.hypot(dx1, dy1))
        a1 = float((np.degrees(np.arctan2(dy1, dx1)) + 360) % 360)

        rows.append([
            side0, t0, tc, tc - t0,
            x0, y0, xc, yc,
            dx1, dy1, L1, a1, True
        ])

        # Segunda parte
        dx2 = x1 - xc
        dy2 = y1 - yc
        L2 = float(np.hypot(dx2, dy2))
        a2 = float((np.degrees(np.arctan2(dy2, dx2)) + 360) % 360)

        rows.append([
            side1, tc, t1, t1 - tc,
            xc, yc, x1, y1,
            dx2, dy2, L2, a2, True
        ])

        if side0 in ("Claro", "Escuro") and side1 in ("Claro", "Escuro") and side0 != side1:
            events.append({
                "tempo": float(tc),
                "direcao": f"{side0}→{side1}",
                "velocidade": float(full_speed),
            })

    cols = [
        "lado","t_inicial","t_final","dt",
        "x_inicial","y_inicial","x_final","y_final",
        "dx","dy","tamanho_vetor","orientacao_graus","cruzamento"
    ]

    seg = pd.DataFrame(rows, columns=cols)

    if not seg.empty:
        seg["velocidade"] = np.where(
            seg["dt"] > 0,
            seg["tamanho_vetor"] / seg["dt"],
            np.nan
        )

    return seg, pd.DataFrame(events), labels


def transition_metrics_from_events(events):
    if events.empty:
        return {
            "crossing_times": np.array([], dtype=float),
            "crossing_directions": [],
            "crossing_speeds": np.array([], dtype=float),
            "intervals": np.array([], dtype=float),
            "claro_intervals": np.array([], dtype=float),
            "escuro_intervals": np.array([], dtype=float),
        }

    ev = events.sort_values("tempo").reset_index(drop=True)

    times = ev["tempo"].to_numpy(float)
    directions = ev["direcao"].tolist()
    speeds = ev["velocidade"].to_numpy(float)

    intervals = np.diff(times) if len(times) >= 2 else np.array([], dtype=float)

    claro_intervals = []
    escuro_intervals = []

    for i in range(len(times) - 1):
        duration = times[i + 1] - times[i]

        if directions[i] == "Escuro→Claro":
            claro_intervals.append(duration)
        elif directions[i] == "Claro→Escuro":
            escuro_intervals.append(duration)

    return {
        "crossing_times": times,
        "crossing_directions": directions,
        "crossing_speeds": speeds,
        "intervals": np.asarray(intervals, dtype=float),
        "claro_intervals": np.asarray(claro_intervals, dtype=float),
        "escuro_intervals": np.asarray(escuro_intervals, dtype=float),
    }



def classify_stopped_events(
    stopped_events,
    full_df,
    selected_time,
    regions=None,
    region_to_side=None,
):
    """
    Classifica episódios de e.parado como Claro/Escuro pela posição
    interpolada no ponto médio temporal do episódio.

    Os tempos de e.parado são convertidos para a mesma origem temporal
    usada no app. Episódios iniciados até selected_time são incluídos;
    se ainda estiverem em andamento, tf é truncado no tempo selecionado.
    """
    if stopped_events is None or stopped_events.empty:
        return pd.DataFrame(
            columns=[
                "ti", "tf", "duracao", "tempo_medio",
                "x", "y", "lado"
            ]
        )

    # origem temporal original antes de prepare_data zerar o tempo
    t0_original = float(full_df["tempo_original"].iloc[0])

    ev = stopped_events.copy()
    ev["ti_rel"] = ev["ti"] - t0_original
    ev["tf_rel"] = ev["tf"] - t0_original

    # Se e.parado já estiver em tempo relativo, a subtração acima pode
    # deslocar incorretamente. Detecta isso pela faixa de tempo.
    total_duration = float(full_df["tempo"].iloc[-1])
    if (
        len(ev)
        and (
            ev["tf_rel"].max() < -1e-6
            or ev["ti_rel"].min() > total_duration + 1
        )
    ):
        ev["ti_rel"] = ev["ti"]
        ev["tf_rel"] = ev["tf"]

    ev = ev[
        (ev["ti_rel"] <= selected_time) &
        (ev["tf_rel"] >= 0)
    ].copy()

    if ev.empty:
        return pd.DataFrame(
            columns=[
                "ti", "tf", "duracao", "tempo_medio",
                "x", "y", "lado"
            ]
        )

    ev["ti_clip"] = ev["ti_rel"].clip(lower=0)
    ev["tf_clip"] = ev["tf_rel"].clip(upper=selected_time)
    ev = ev[ev["tf_clip"] >= ev["ti_clip"]].copy()

    t_mid = (
        ev["ti_clip"].to_numpy(float) +
        ev["tf_clip"].to_numpy(float)
    ) / 2.0

    t_series = full_df["tempo"].to_numpy(float)
    x_orig_series = full_df["x_original"].to_numpy(float)
    x_norm_series = full_df["x"].to_numpy(float)
    y_series = full_df["y"].to_numpy(float)

    x_orig_mid = np.interp(t_mid, t_series, x_orig_series)
    x_norm_mid = np.interp(t_mid, t_series, x_norm_series)
    y_mid = np.interp(t_mid, t_series, y_series)

    sides = []

    for xo, xn, yy in zip(x_orig_mid, x_norm_mid, y_mid):
        if regions and region_to_side:
            reg = classify_region(float(xo), float(yy), regions)
            side = region_to_side.get(reg, "Fora")
        else:
            if xn < 0:
                side = "Claro"
            elif xn > 0:
                side = "Escuro"
            else:
                side = "Fronteira"

        sides.append(side)

    out = pd.DataFrame({
        "ti": ev["ti_clip"].to_numpy(float),
        "tf": ev["tf_clip"].to_numpy(float),
        "duracao": (
            ev["tf_clip"].to_numpy(float) -
            ev["ti_clip"].to_numpy(float)
        ),
        "tempo_medio": t_mid,
        "x": x_norm_mid,
        "y": y_mid,
        "lado": sides,
    })

    return out


def circular_mean_deg(angles_deg, weights=None):
    a = np.deg2rad(np.asarray(angles_deg, float))
    if len(a) == 0: return np.nan
    w = np.ones_like(a) if weights is None else np.asarray(weights, float)
    ok = np.isfinite(a) & np.isfinite(w)
    if not np.any(ok): return np.nan
    s = np.sum(w[ok]*np.sin(a[ok])); c = np.sum(w[ok]*np.cos(a[ok]))
    if np.isclose(s,0) and np.isclose(c,0): return np.nan
    return (np.degrees(np.arctan2(s,c))+360)%360


def covariance_ellipse(dx, dy):
    pts = np.column_stack([np.asarray(dx,float), np.asarray(dy,float)])
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    if len(pts) < 3: return None
    cov = np.cov(pts, rowvar=False, ddof=1)
    if not np.all(np.isfinite(cov)): return None
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals = np.maximum(vals[order],0); vecs = vecs[:,order]
    scale = np.sqrt(5.991)
    major, minor = scale*np.sqrt(vals[0]), scale*np.sqrt(vals[1])
    vx,vy = vecs[:,0]
    angle = (np.degrees(np.arctan2(vy,vx))+180)%180
    th = np.linspace(0,2*np.pi,361)
    el = vecs @ np.diag([major,minor]) @ np.vstack([np.cos(th),np.sin(th)])
    return {'major':major,'minor':minor,'directionality':np.inf if minor==0 else major/minor,'angle':angle,'x':el[0],'y':el[1]}


def split_segments(df):
    t,x,y = [df[c].to_numpy(float) for c in ('tempo','x','y')]
    rows=[]
    for i in range(len(df)-1):
        t0,t1=t[i],t[i+1]; x0,x1=x[i],x[i+1]; y0,y1=y[i],y[i+1]
        dt=t1-t0
        if dt<=0: continue
        dx=x1-x0; dy=y1-y0
        if (x0>=0 and x1>=0) or (x0<=0 and x1<=0):
            side='Escuro' if x0>=0 and x1>=0 else 'Claro'
            L=np.hypot(dx,dy); ang=(np.degrees(np.arctan2(dy,dx))+360)%360
            rows.append([side,t0,t1,dt,x0,y0,x1,y1,dx,dy,L,ang,False])
        else:
            f=float(np.clip(-x0/(x1-x0),0,1)); tc=t0+f*dt; yc=y0+f*dy
            for side,ta,tb,xa,ya,xb,yb in [
                ('Escuro' if x0>0 else 'Claro', t0,tc,x0,y0,0.0,yc),
                ('Escuro' if x1>0 else 'Claro', tc,t1,0.0,yc,x1,y1),
            ]:
                ddx=xb-xa; ddy=yb-ya; L=np.hypot(ddx,ddy); ang=(np.degrees(np.arctan2(ddy,ddx))+360)%360
                rows.append([side,ta,tb,tb-ta,xa,ya,xb,yb,ddx,ddy,L,ang,True])
    cols=['lado','t_inicial','t_final','dt','x_inicial','y_inicial','x_final','y_final','dx','dy','tamanho_vetor','orientacao_graus','cruzamento']
    seg=pd.DataFrame(rows, columns=cols)
    if not seg.empty:
        seg['velocidade']=np.where(seg['dt']>0,seg['tamanho_vetor']/seg['dt'],np.nan)
    return seg


def summarize(seg,label):
    if seg.empty:
        return {'regiao':label,'tempo_total':0.0,'distancia_total':0.0,'velocidade_media':np.nan,'velocidade_dp':np.nan,'vetor_medio':np.nan,'vetor_dp':np.nan,'orientacao_media':np.nan,'semieixo_maior':np.nan,'semieixo_menor':np.nan,'indice_direcionalidade':np.nan,'orientacao_elipse':np.nan,'n_segmentos':0}
    T=float(seg['dt'].sum()); D=float(seg['tamanho_vetor'].sum())
    ell=covariance_ellipse(seg['dx'],seg['dy'])
    return {
        'regiao':label,'tempo_total':T,'distancia_total':D,
        'velocidade_media':D/T if T>0 else np.nan,
        'velocidade_dp':float(seg['velocidade'].std(ddof=1)) if len(seg)>1 else np.nan,
        'vetor_medio':float(seg['tamanho_vetor'].mean()),
        'vetor_dp':float(seg['tamanho_vetor'].std(ddof=1)) if len(seg)>1 else np.nan,
        'orientacao_media':circular_mean_deg(seg['orientacao_graus'],seg['tamanho_vetor']),
        'semieixo_maior':ell['major'] if ell else np.nan,
        'semieixo_menor':ell['minor'] if ell else np.nan,
        'indice_direcionalidade':ell['directionality'] if ell else np.nan,
        'orientacao_elipse':ell['angle'] if ell else np.nan,
        'n_segmentos':int(len(seg)),
    }



def transition_metrics(df):
    """
    Calcula os tempos exatos de cruzamento de X=0 por interpolação linear.

    Retorna:
      - tempos de cruzamento
      - intervalos entre cruzamentos consecutivos
      - duração das permanências completas em Claro e Escuro
        entre transições consecutivas
    """
    t = df["tempo"].to_numpy(float)
    x = df["x"].to_numpy(float)

    crossing_times = []
    crossing_directions = []
    crossing_speeds = []

    for i in range(len(df) - 1):
        t0, t1 = t[i], t[i + 1]
        x0, x1 = x[i], x[i + 1]

        if t1 <= t0:
            continue

        # Cruzamento real entre lados opostos
        if (x0 < 0 and x1 > 0) or (x0 > 0 and x1 < 0):
            f = -x0 / (x1 - x0)
            tc = t0 + f * (t1 - t0)

            crossing_times.append(float(tc))

            # Velocidade do vetor original que contém o cruzamento
            y0 = float(df["y"].iloc[i])
            y1 = float(df["y"].iloc[i + 1])
            distance = float(np.hypot(x1 - x0, y1 - y0))
            crossing_speeds.append(distance / (t1 - t0))

            if x0 < 0 and x1 > 0:
                crossing_directions.append("Claro→Escuro")
            else:
                crossing_directions.append("Escuro→Claro")

    crossing_times = np.asarray(crossing_times, dtype=float)

    if len(crossing_times) >= 2:
        intervals = np.diff(crossing_times)
    else:
        intervals = np.asarray([], dtype=float)

    # Permanências completas entre transições sucessivas.
    # O lado ocupado entre duas transições é determinado pela direção
    # da primeira transição.
    claro_intervals = []
    escuro_intervals = []

    if len(crossing_times) >= 2:
        for i in range(len(crossing_times) - 1):
            duration = crossing_times[i + 1] - crossing_times[i]

            if crossing_directions[i] == "Claro→Escuro":
                escuro_intervals.append(duration)
            else:
                claro_intervals.append(duration)

    return {
        "crossing_times": crossing_times,
        "crossing_directions": crossing_directions,
        "crossing_speeds": np.asarray(crossing_speeds, dtype=float),
        "intervals": np.asarray(intervals, dtype=float),
        "claro_intervals": np.asarray(claro_intervals, dtype=float),
        "escuro_intervals": np.asarray(escuro_intervals, dtype=float),
    }


def count_crossings(df):
    x=df['x'].to_numpy(float); s=np.sign(x)
    for i in range(1,len(s)):
        if s[i]==0: s[i]=s[i-1]
    for i in range(len(s)-2,-1,-1):
        if s[i]==0: s[i]=s[i+1]
    return int(np.sum(s[1:]*s[:-1]<0))


def fmt(v,d=3):
    return '—' if v is None or not np.isfinite(v) else f'{v:.{d}f}'

uploaded=st.file_uploader('Selecione o arquivo', type=['csv','mat'])
if uploaded is None:
    st.info('Carregue um CSV ou .mat para iniciar.'); st.stop()
try:
    regions = []
    processed_region = None
    stopped_events = pd.DataFrame(columns=['ti', 'tf'])

    if uploaded.name.lower().endswith('.mat'):
        raw, regions, processed_region, stopped_events = extract_from_mat(uploaded)
        ftype='MATLAB'
    else:
        raw=extract_from_csv(uploaded)
        ftype='CSV'

    df,x0orig=prepare_data(raw)

except Exception as exc:
    st.error(str(exc)); st.stop()

st.success(
    f'{ftype} carregado: {len(df)} amostras. '
    f'X inicial original={x0orig:.4f}; novo X inicial=0.'
)

if regions:
    st.info(
        f'Foram encontrados {len(regions)} polígonos em e.areaint. '
        'Eles serão usados para classificar geometricamente as regiões.'
    )
else:
    st.info(
        'O arquivo não possui polígonos utilizáveis; será usada a separação X<0 / X>0.'
    )

tmax=float(df['tempo'].iloc[-1])
selected=st.slider('Mostrar trajetória até',0.0,tmax,tmax,step=max(tmax/1500,0.01),format='%.2f s')
cur=df[df['tempo']<=selected].copy()
if len(cur)<2:
    st.warning('Aumente o tempo para incluir pelo menos duas coordenadas.'); st.stop()
if regions and len(regions) >= 2:
    st.subheader('Identificação das regiões')

    r1 = regions[0]['name']
    r2 = regions[1]['name']

    mapping_choice = st.radio(
        'Qual região corresponde ao claro e ao escuro?',
        [
            f'{r1} = Claro | {r2} = Escuro',
            f'{r1} = Escuro | {r2} = Claro',
        ],
        horizontal=True,
    )

    if mapping_choice.startswith(f'{r1} = Claro'):
        region_to_side = {r1: 'Claro', r2: 'Escuro'}
    else:
        region_to_side = {r1: 'Escuro', r2: 'Claro'}

    seg, region_events, point_region_labels = split_segments_regions(
        cur, regions, region_to_side
    )
    transitions = transition_metrics_from_events(region_events)
    cross = len(transitions['crossing_times'])

else:
    seg=split_segments(cur)
    transitions = transition_metrics(cur)
    cross=count_crossings(cur)
    region_to_side = {}

if seg.empty:
    st.warning('Não foi possível formar segmentos válidos.'); st.stop()

# Métricas específicas de claro e escuro
clear=seg[seg['lado']=='Claro'].copy()
dark=seg[seg['lado']=='Escuro'].copy()

# Campo total usa todos os segmentos que pertencem a claro/escuro.
valid_seg = seg[seg['lado'].isin(['Claro','Escuro'])].copy()
if valid_seg.empty:
    valid_seg = seg.copy()

sall=summarize(valid_seg,'Campo total')
sc=summarize(clear,'Claro')
sd=summarize(dark,'Escuro')

T=float(cur['tempo'].iloc[-1]-cur['tempo'].iloc[0])
D=float(valid_seg['tamanho_vetor'].sum())
mean_speed=D/T if T>0 else np.nan
speed_sd=float(valid_seg['velocidade'].std(ddof=1)) if len(valid_seg)>1 else np.nan

inter_transition = transitions["intervals"]
mean_transition_interval = (
    float(np.mean(inter_transition)) if len(inter_transition) else np.nan
)
sd_transition_interval = (
    float(np.std(inter_transition, ddof=1))
    if len(inter_transition) > 1 else np.nan
)
median_transition_interval = (
    float(np.median(inter_transition)) if len(inter_transition) else np.nan
)

clear_stays = transitions["claro_intervals"]
dark_stays = transitions["escuro_intervals"]

mean_clear_stay = (
    float(np.mean(clear_stays)) if len(clear_stays) else np.nan
)
sd_clear_stay = (
    float(np.std(clear_stays, ddof=1))
    if len(clear_stays) > 1 else np.nan
)

mean_dark_stay = (
    float(np.mean(dark_stays)) if len(dark_stays) else np.nan
)
sd_dark_stay = (
    float(np.std(dark_stays, ddof=1))
    if len(dark_stays) > 1 else np.nan
)

st.subheader('Trajetória')
fig=go.Figure()
fig.add_trace(go.Scatter(
    x=cur['x'],y=cur['y'],mode='lines',
    name='Trajetória',line=dict(width=2)
))

if regions and len(regions) >= 2:
    # Pontos classificados pelos polígonos reais
    labels_now = [
        classify_region(
            float(cur['x_original'].iloc[i]),
            float(cur['y'].iloc[i]),
            regions
        )
        for i in range(len(cur))
    ]

    labels_now = np.asarray(labels_now, dtype=object)

    for reg_name, side_name in region_to_side.items():
        mask = labels_now == reg_name
        fig.add_trace(go.Scatter(
            x=cur.loc[mask,'x'],
            y=cur.loc[mask,'y'],
            mode='markers',
            name=side_name,
            marker=dict(size=4,opacity=.40)
        ))

    # Desenha limites convertidos para cm e deslocados pela origem X inicial
    for reg in regions:
        rx = np.asarray(reg['x'], dtype=float) - x0orig
        ry = np.asarray(reg['y'], dtype=float)

        fig.add_trace(go.Scatter(
            x=np.r_[rx,rx[0]],
            y=np.r_[ry,ry[0]],
            mode='lines',
            name=f"Limite {reg['name']}",
            line=dict(width=3,dash='dash')
        ))

else:
    fig.add_trace(go.Scatter(
        x=cur.loc[cur['x']<=0,'x'],
        y=cur.loc[cur['x']<=0,'y'],
        mode='markers',name='Claro',
        marker=dict(size=4,opacity=.4)
    ))
    fig.add_trace(go.Scatter(
        x=cur.loc[cur['x']>=0,'x'],
        y=cur.loc[cur['x']>=0,'y'],
        mode='markers',name='Escuro',
        marker=dict(size=4,opacity=.4)
    ))
    fig.add_vline(
        x=0,line_width=2,line_dash='dash',
        annotation_text='X = 0'
    )
fig.add_trace(go.Scatter(x=[cur['x'].iloc[0]],y=[cur['y'].iloc[0]],mode='markers',name='Início',marker=dict(size=11)))
fig.add_trace(go.Scatter(x=[cur['x'].iloc[-1]],y=[cur['y'].iloc[-1]],mode='markers',name='Atual',marker=dict(size=11,symbol='diamond')))
fig.update_layout(xaxis_title='X normalizado',yaxis_title='Y',height=600,legend=dict(orientation='h'))
fig.update_yaxes(scaleanchor='x',scaleratio=1)
st.plotly_chart(fig,use_container_width=True)


# ------------------------------------------------------------
# Episódios de imobilidade / parado
# ------------------------------------------------------------
classified_stops = classify_stopped_events(
    stopped_events=stopped_events,
    full_df=df,
    selected_time=selected,
    regions=regions if regions else None,
    region_to_side=region_to_side if regions else None,
)

clear_stops = classified_stops[
    classified_stops["lado"] == "Claro"
].copy()

dark_stops = classified_stops[
    classified_stops["lado"] == "Escuro"
].copy()

n_clear_stops = len(clear_stops)
n_dark_stops = len(dark_stops)

time_clear_stopped = (
    float(clear_stops["duracao"].sum())
    if n_clear_stops else 0.0
)
time_dark_stopped = (
    float(dark_stops["duracao"].sum())
    if n_dark_stops else 0.0
)

mean_clear_stop = (
    float(clear_stops["duracao"].mean())
    if n_clear_stops else np.nan
)
mean_dark_stop = (
    float(dark_stops["duracao"].mean())
    if n_dark_stops else np.nan
)

st.subheader("Episódios em que o animal ficou parado")

cols = st.columns(4)
cols[0].metric("Nº episódios parado — Claro", str(n_clear_stops))
cols[1].metric("Nº episódios parado — Escuro", str(n_dark_stops))
cols[2].metric(
    "Tempo total parado — Claro",
    f"{fmt(time_clear_stopped, 2)} s"
)
cols[3].metric(
    "Tempo total parado — Escuro",
    f"{fmt(time_dark_stopped, 2)} s"
)

cols = st.columns(2)
cols[0].metric(
    "Duração média episódio — Claro",
    f"{fmt(mean_clear_stop, 2)} s"
)
cols[1].metric(
    "Duração média episódio — Escuro",
    f"{fmt(mean_dark_stop, 2)} s"
)

if stopped_events.empty:
    st.caption(
        "Este arquivo não contém episódios em e.parado; "
        "por isso as métricas de imobilidade não estão disponíveis."
    )
else:
    st.caption(
        "Os episódios são obtidos diretamente de e.parado.ti/e.parado.tf. "
        "Cada episódio é classificado pela posição interpolada do animal "
        "no ponto médio temporal do intervalo."
    )

with st.expander("Ver episódios de imobilidade"):
    if classified_stops.empty:
        st.info("Nenhum episódio disponível no trecho temporal selecionado.")
    else:
        st.dataframe(
            classified_stops.round(4),
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Baixar episódios de imobilidade em CSV",
            data=classified_stops.to_csv(index=False).encode("utf-8"),
            file_name="episodios_parado_claro_escuro.csv",
            mime="text/csv",
        )

st.subheader('Métricas gerais')
cols=st.columns(4)
cols[0].metric('Distância total',fmt(D)); cols[1].metric('Tempo total',f'{fmt(T,2)} s'); cols[2].metric('Velocidade média',fmt(mean_speed)); cols[3].metric('DP velocidade',fmt(speed_sd))
cols=st.columns(4)
cols[0].metric('Tempo claro',f"{fmt(sc['tempo_total'],2)} s"); cols[1].metric('Tempo escuro',f"{fmt(sd['tempo_total'],2)} s"); cols[2].metric('% tempo claro',f"{fmt(100*sc['tempo_total']/T if T>0 else np.nan,1)}%"); cols[3].metric('% tempo escuro',f"{fmt(100*sd['tempo_total']/T if T>0 else np.nan,1)}%")
cols=st.columns(3)
cols[0].metric('Distância claro',fmt(sc['distancia_total'])); cols[1].metric('Distância escuro',fmt(sd['distancia_total'])); cols[2].metric('Cruzamentos',str(cross))

st.subheader('Dinâmica das transições')

cols=st.columns(3)
cols[0].metric(
    'Intervalo médio entre transições',
    f'{fmt(mean_transition_interval,2)} s'
)
cols[1].metric(
    'DP do intervalo entre transições',
    f'{fmt(sd_transition_interval,2)} s'
)
cols[2].metric(
    'Mediana entre transições',
    f'{fmt(median_transition_interval,2)} s'
)

cols=st.columns(4)
cols[0].metric(
    'Permanência média no claro',
    f'{fmt(mean_clear_stay,2)} s'
)
cols[1].metric(
    'DP permanência no claro',
    f'{fmt(sd_clear_stay,2)} s'
)
cols[2].metric(
    'Permanência média no escuro',
    f'{fmt(mean_dark_stay,2)} s'
)
cols[3].metric(
    'DP permanência no escuro',
    f'{fmt(sd_dark_stay,2)} s'
)

st.caption(
    'O intervalo entre transições é o tempo entre duas transições consecutivas entre claro e escuro. '
    'Quando há polígonos em e.areaint, o cruzamento é estimado na fronteira real por interpolação linear.'
)

comp=pd.DataFrame([sall,sc,sd])
comp=comp[['regiao','tempo_total','distancia_total','velocidade_media','velocidade_dp','vetor_medio','vetor_dp','orientacao_media','semieixo_maior','semieixo_menor','indice_direcionalidade','orientacao_elipse','n_segmentos']]
comp.columns=['Região','Tempo total','Distância total','Velocidade média','DP velocidade','Tamanho médio vetor','DP tamanho vetor','Orientação média (°)','Semieixo maior','Semieixo menor','Índice direcionalidade','Orientação elipse (°)','N segmentos']
st.subheader('Comparação entre regiões')
st.dataframe(comp.round(4),use_container_width=True,hide_index=True)

st.subheader('Índice de direcionalidade')

cols = st.columns(3)
cols[0].metric(
    'Campo total',
    fmt(sall['indice_direcionalidade'], 3)
)
cols[1].metric(
    'Claro',
    fmt(sc['indice_direcionalidade'], 3)
)
cols[2].metric(
    'Escuro',
    fmt(sd['indice_direcionalidade'], 3)
)

cols = st.columns(3)
cols[0].metric(
    'Semieixos — Campo total',
    f"{fmt(sall['semieixo_maior'],3)} / {fmt(sall['semieixo_menor'],3)}"
)
cols[1].metric(
    'Semieixos — Claro',
    f"{fmt(sc['semieixo_maior'],3)} / {fmt(sc['semieixo_menor'],3)}"
)
cols[2].metric(
    'Semieixos — Escuro',
    f"{fmt(sd['semieixo_maior'],3)} / {fmt(sd['semieixo_menor'],3)}"
)

st.caption(
    'Índice de direcionalidade = semieixo maior / semieixo menor. '
    'Valores próximos de 1 indicam distribuição mais isotrópica; '
    'valores maiores indicam maior anisotropia direcional.'
)

st.subheader('Velocidade nas transições')

transition_speed_df = pd.DataFrame({
    "direcao": transitions["crossing_directions"],
    "velocidade": transitions["crossing_speeds"],
})

if not transition_speed_df.empty:
    ce = transition_speed_df.loc[
        transition_speed_df["direcao"] == "Claro→Escuro",
        "velocidade"
    ]
    ec = transition_speed_df.loc[
        transition_speed_df["direcao"] == "Escuro→Claro",
        "velocidade"
    ]

    cols = st.columns(4)
    cols[0].metric(
        "Vel. média Claro→Escuro",
        fmt(float(ce.mean())) if len(ce) else "—"
    )
    cols[1].metric(
        "DP Claro→Escuro",
        fmt(float(ce.std(ddof=1))) if len(ce) > 1 else "—"
    )
    cols[2].metric(
        "Vel. média Escuro→Claro",
        fmt(float(ec.mean())) if len(ec) else "—"
    )
    cols[3].metric(
        "DP Escuro→Claro",
        fmt(float(ec.std(ddof=1))) if len(ec) > 1 else "—"
    )

st.caption(
    "A velocidade de transição corresponde à velocidade do vetor entre as duas "
    "amostras consecutivas que atravessam X = 0."
)

st.subheader('Velocidade ao longo do tempo')
figv=go.Figure()
for name,dsub in [('Claro',clear),('Escuro',dark)]:
    if not dsub.empty:
        figv.add_trace(go.Scatter(
            x=dsub['t_final'],
            y=dsub['velocidade'],
            mode='markers',
            name=name,
            marker=dict(size=4,opacity=.45)
        ))

# Destacar as velocidades exatamente nos vetores que cruzam X = 0
cross_t = transitions["crossing_times"]
cross_dir = transitions["crossing_directions"]
cross_v = transitions["crossing_speeds"]

for direction, symbol in [
    ("Claro→Escuro", "triangle-up"),
    ("Escuro→Claro", "triangle-down"),
]:
    mask = np.array([d == direction for d in cross_dir], dtype=bool)
    if np.any(mask):
        figv.add_trace(
            go.Scatter(
                x=cross_t[mask],
                y=cross_v[mask],
                mode="markers",
                name=direction,
                marker=dict(size=13, symbol=symbol, line=dict(width=1.5)),
                hovertemplate=(
                    "Transição: " + direction +
                    "<br>Tempo=%{x:.2f} s"
                    "<br>Velocidade=%{y:.3f}<extra></extra>"
                ),
            )
        )

figv.update_layout(
    xaxis_title='Tempo (s)',
    yaxis_title='Velocidade',
    height=460,
    legend=dict(orientation='h')
)
st.plotly_chart(figv,use_container_width=True)

st.subheader('Distribuição angular dos vetores — gráfico polar')
bw=st.select_slider(
    'Largura dos setores angulares',
    options=[5,10,15,20,30,45],
    value=15,
    format_func=lambda x:f'{x}°'
)
polar_mode=st.radio(
    'Representação radial',
    options=['Número de vetores','Percentual de vetores'],
    horizontal=True
)

bins=np.arange(0,360+bw,bw)
centers=(bins[:-1]+bins[1:])/2

figa=go.Figure()

for name,dsub in [('Claro',clear),('Escuro',dark)]:
    if dsub.empty:
        continue

    h,_=np.histogram(dsub['orientacao_graus'],bins=bins)

    if polar_mode == 'Percentual de vetores':
        total=h.sum()
        radial=(100.0*h/total) if total>0 else h.astype(float)
        hover='Orientação=%{theta:.1f}°<br>Percentual=%{r:.2f}%<extra>'+name+'</extra>'
        radial_title='Percentual (%)'
    else:
        radial=h
        hover='Orientação=%{theta:.1f}°<br>Vetores=%{r}<extra>'+name+'</extra>'
        radial_title='Número de vetores'

    figa.add_trace(
        go.Barpolar(
            r=radial,
            theta=centers,
            width=[bw*0.90]*len(centers),
            name=name,
            opacity=0.62,
            hovertemplate=hover
        )
    )

figa.update_layout(
    height=620,
    margin=dict(l=30,r=30,t=40,b=30),
    polar=dict(
        angularaxis=dict(
            direction='counterclockwise',
            rotation=0,
            tickmode='array',
            tickvals=[0,45,90,135,180,225,270,315],
            ticktext=['0°','45°','90°','135°','180°','225°','270°','315°']
        ),
        radialaxis=dict(
            title=radial_title,
            angle=90
        )
    ),
    barmode='overlay',
    legend=dict(orientation='h')
)

st.plotly_chart(figa,use_container_width=True)

st.caption(
    '0° = deslocamento para +X; 90° = +Y; 180° = −X; 270° = −Y. '
    'Os setores mostram a frequência angular dos vetores separadamente para Claro e Escuro.'
)


st.subheader("Tempo, distância e velocidade por orientação")

orientation_bin = st.select_slider(
    "Setores angulares para análise por orientação",
    options=[15, 30, 45, 60, 90],
    value=45,
    format_func=lambda x: f"{x}°"
)

def orientation_summary(seg_df, bin_width):
    if seg_df.empty:
        return pd.DataFrame()

    edges = np.arange(0, 360 + bin_width, bin_width)
    labels = []

    for a, b in zip(edges[:-1], edges[1:]):
        labels.append(f"{int(a)}–{int(b)}°")

    temp = seg_df.copy()
    temp["setor"] = pd.cut(
        temp["orientacao_graus"],
        bins=edges,
        labels=labels,
        include_lowest=True,
        right=False,
    )

    out = (
        temp.groupby("setor", observed=False)
        .agg(
            tempo_total=("dt", "sum"),
            distancia_total=("tamanho_vetor", "sum"),
            n_vetores=("tamanho_vetor", "size"),
            tamanho_medio_vetor=("tamanho_vetor", "mean"),
        )
        .reset_index()
    )

    out["velocidade_media"] = np.where(
        out["tempo_total"] > 0,
        out["distancia_total"] / out["tempo_total"],
        np.nan,
    )

    out["percentual_tempo"] = (
        100.0 * out["tempo_total"] / out["tempo_total"].sum()
        if out["tempo_total"].sum() > 0
        else np.nan
    )

    out["percentual_distancia"] = (
        100.0 * out["distancia_total"] / out["distancia_total"].sum()
        if out["distancia_total"].sum() > 0
        else np.nan
    )

    return out


ori_total = orientation_summary(seg, orientation_bin)
ori_claro = orientation_summary(clear, orientation_bin)
ori_escuro = orientation_summary(dark, orientation_bin)

tab1, tab2, tab3 = st.tabs(["Campo total", "Claro", "Escuro"])

for tab, title, ori in [
    (tab1, "Campo total", ori_total),
    (tab2, "Claro", ori_claro),
    (tab3, "Escuro", ori_escuro),
]:
    with tab:
        if ori.empty:
            st.info("Sem dados suficientes.")
            continue

        c1, c2 = st.columns(2)

        with c1:
            fig_time_ori = go.Figure(
                go.Barpolar(
                    r=ori["tempo_total"],
                    theta=[
                        (i * orientation_bin) + orientation_bin / 2
                        for i in range(len(ori))
                    ],
                    width=[orientation_bin * 0.9] * len(ori),
                    hovertemplate=(
                        "Setor=%{customdata}"
                        "<br>Tempo=%{r:.3f} s<extra></extra>"
                    ),
                    customdata=ori["setor"].astype(str),
                    name=title,
                )
            )

            fig_time_ori.update_layout(
                title=f"Tempo por orientação — {title}",
                height=450,
                polar=dict(
                    angularaxis=dict(
                        direction="counterclockwise",
                        rotation=0,
                        tickvals=[0,45,90,135,180,225,270,315],
                        ticktext=["0°","45°","90°","135°","180°","225°","270°","315°"]
                    ),
                    radialaxis=dict(title="Tempo (s)")
                ),
                margin=dict(l=20, r=20, t=55, b=20)
            )

            st.plotly_chart(fig_time_ori, use_container_width=True)

        with c2:
            fig_dist_ori = go.Figure(
                go.Barpolar(
                    r=ori["distancia_total"],
                    theta=[
                        (i * orientation_bin) + orientation_bin / 2
                        for i in range(len(ori))
                    ],
                    width=[orientation_bin * 0.9] * len(ori),
                    hovertemplate=(
                        "Setor=%{customdata}"
                        "<br>Distância=%{r:.3f}<extra></extra>"
                    ),
                    customdata=ori["setor"].astype(str),
                    name=title,
                )
            )

            fig_dist_ori.update_layout(
                title=f"Distância por orientação — {title}",
                height=450,
                polar=dict(
                    angularaxis=dict(
                        direction="counterclockwise",
                        rotation=0,
                        tickvals=[0,45,90,135,180,225,270,315],
                        ticktext=["0°","45°","90°","135°","180°","225°","270°","315°"]
                    ),
                    radialaxis=dict(title="Distância")
                ),
                margin=dict(l=20, r=20, t=55, b=20)
            )

            st.plotly_chart(fig_dist_ori, use_container_width=True)

        fig_vel_ori = go.Figure(
            go.Bar(
                x=ori["setor"].astype(str),
                y=ori["velocidade_media"],
                customdata=np.column_stack([
                    ori["tempo_total"],
                    ori["distancia_total"],
                    ori["n_vetores"]
                ]),
                hovertemplate=(
                    "Setor=%{x}"
                    "<br>Velocidade média=%{y:.3f}"
                    "<br>Tempo=%{customdata[0]:.3f} s"
                    "<br>Distância=%{customdata[1]:.3f}"
                    "<br>N vetores=%{customdata[2]}<extra></extra>"
                )
            )
        )

        fig_vel_ori.update_layout(
            title=f"Velocidade média por orientação — {title}",
            xaxis_title="Setor angular",
            yaxis_title="Velocidade média",
            height=420,
            margin=dict(l=20, r=20, t=55, b=20),
        )

        st.plotly_chart(fig_vel_ori, use_container_width=True)

        display_ori = ori.copy()
        display_ori.columns = [
            "Setor angular",
            "Tempo total",
            "Distância total",
            "N vetores",
            "Tamanho médio do vetor",
            "Velocidade média",
            "% do tempo",
            "% da distância",
        ]

        st.dataframe(
            display_ori.round(4),
            use_container_width=True,
            hide_index=True,
        )

st.caption(
    "Para cada setor angular, o tempo é a soma dos Δt dos vetores naquele setor; "
    "a distância é a soma dos comprimentos dos vetores; e a velocidade média é "
    "calculada como distância total do setor / tempo total do setor."
)


st.subheader("Entropia espacial e direcional")

def shannon_entropy_normalized(probabilities):
    p = np.asarray(probabilities, dtype=float)
    p = p[np.isfinite(p) & (p > 0)]

    if len(p) <= 1:
        return 0.0, 0.0

    p = p / p.sum()

    H = -np.sum(p * np.log(p))
    Hmax = np.log(len(p))

    Hnorm = H / Hmax if Hmax > 0 else 0.0

    return float(H), float(Hnorm)


def spatial_entropy_from_points(df_points, cell_size):
    """
    Calcula entropia de ocupação espacial usando uma grade regular.
    O peso é baseado na frequência de amostras em cada célula.
    """
    if df_points.empty:
        return np.nan, np.nan, pd.DataFrame()

    x = df_points["x"].to_numpy(float)
    y = df_points["y"].to_numpy(float)

    if len(x) < 2:
        return np.nan, np.nan, pd.DataFrame()

    xmin, xmax = np.nanmin(x), np.nanmax(x)
    ymin, ymax = np.nanmin(y), np.nanmax(y)

    # Evita grade degenerada
    if np.isclose(xmin, xmax):
        xmax = xmin + cell_size
    if np.isclose(ymin, ymax):
        ymax = ymin + cell_size

    x_edges = np.arange(xmin, xmax + cell_size, cell_size)
    y_edges = np.arange(ymin, ymax + cell_size, cell_size)

    if len(x_edges) < 2:
        x_edges = np.array([xmin, xmin + cell_size])
    if len(y_edges) < 2:
        y_edges = np.array([ymin, ymin + cell_size])

    hist, xe, ye = np.histogram2d(
        x,
        y,
        bins=[x_edges, y_edges]
    )

    counts = hist.ravel()
    occupied = counts[counts > 0]

    if occupied.size == 0:
        return np.nan, np.nan, pd.DataFrame()

    probs = occupied / occupied.sum()
    H, Hnorm = shannon_entropy_normalized(probs)

    # Tabela para heatmap
    centers_x = (xe[:-1] + xe[1:]) / 2
    centers_y = (ye[:-1] + ye[1:]) / 2

    heat = pd.DataFrame(
        hist.T,
        index=centers_y,
        columns=centers_x,
    )

    return H, Hnorm, heat


def directional_entropy(seg_df, bin_width):
    """
    Entropia da distribuição dos ângulos dos vetores.
    """
    if seg_df.empty:
        return np.nan, np.nan

    angles = seg_df["orientacao_graus"].to_numpy(float)

    bins = np.arange(0, 360 + bin_width, bin_width)
    hist, _ = np.histogram(angles, bins=bins)

    if hist.sum() == 0:
        return np.nan, np.nan

    probs = hist[hist > 0] / hist.sum()

    return shannon_entropy_normalized(probs)


# Tamanho de célula espacial
cell_size = st.select_slider(
    "Tamanho da célula para entropia espacial",
    options=[0.25, 0.5, 1.0, 2.0],
    value=0.5,
    format_func=lambda x: f"{x:.2f}"
)

# Largura angular independente da visualização polar anterior
entropy_angle_bin = st.select_slider(
    "Largura dos setores para entropia angular",
    options=[10, 15, 20, 30, 45, 60],
    value=30,
    format_func=lambda x: f"{x}°"
)

# Pontos por região
if regions and len(regions) >= 2:
    labels_entropy = np.asarray([
        classify_region(
            float(cur["x_original"].iloc[i]),
            float(cur["y"].iloc[i]),
            regions
        )
        for i in range(len(cur))
    ], dtype=object)

    clear_region_names = [
        k for k, v in region_to_side.items() if v == "Claro"
    ]
    dark_region_names = [
        k for k, v in region_to_side.items() if v == "Escuro"
    ]

    clear_points = cur[
        np.isin(labels_entropy, clear_region_names)
    ].copy()

    dark_points = cur[
        np.isin(labels_entropy, dark_region_names)
    ].copy()

else:
    clear_points = cur[cur["x"] <= 0].copy()
    dark_points = cur[cur["x"] >= 0].copy()

total_points = cur.copy()

# Entropias espaciais
Hsp_total, Hspn_total, heat_total = spatial_entropy_from_points(
    total_points, cell_size
)
Hsp_clear, Hspn_clear, heat_clear = spatial_entropy_from_points(
    clear_points, cell_size
)
Hsp_dark, Hspn_dark, heat_dark = spatial_entropy_from_points(
    dark_points, cell_size
)

# Entropias direcionais
Hang_total, Hangn_total = directional_entropy(
    valid_seg, entropy_angle_bin
)
Hang_clear, Hangn_clear = directional_entropy(
    clear, entropy_angle_bin
)
Hang_dark, Hangn_dark = directional_entropy(
    dark, entropy_angle_bin
)

st.markdown("#### Entropia espacial normalizada")

cols = st.columns(3)
cols[0].metric(
    "Campo total",
    fmt(Hspn_total, 3)
)
cols[1].metric(
    "Claro",
    fmt(Hspn_clear, 3)
)
cols[2].metric(
    "Escuro",
    fmt(Hspn_dark, 3)
)

st.caption(
    "Entropia espacial próxima de 0 indica ocupação concentrada em poucas células; "
    "valores próximos de 1 indicam ocupação mais distribuída entre as células visitadas."
)

st.markdown("#### Entropia angular normalizada")

cols = st.columns(3)
cols[0].metric(
    "Campo total",
    fmt(Hangn_total, 3)
)
cols[1].metric(
    "Claro",
    fmt(Hangn_clear, 3)
)
cols[2].metric(
    "Escuro",
    fmt(Hangn_dark, 3)
)

st.caption(
    "Entropia angular próxima de 0 indica movimentos concentrados em poucas direções; "
    "valores próximos de 1 indicam maior diversidade de orientações."
)

# Mapa de ocupação espacial
st.markdown("#### Mapas de ocupação")

tabs_entropy = st.tabs(["Campo total", "Claro", "Escuro"])

for tab, title, heat in [
    (tabs_entropy[0], "Campo total", heat_total),
    (tabs_entropy[1], "Claro", heat_clear),
    (tabs_entropy[2], "Escuro", heat_dark),
]:
    with tab:
        if heat.empty:
            st.info("Sem dados suficientes para o mapa de ocupação.")
        else:
            fig_heat = go.Figure(
                data=go.Heatmap(
                    z=heat.to_numpy(),
                    x=heat.columns.to_numpy(float),
                    y=heat.index.to_numpy(float),
                    colorbar=dict(title="N amostras"),
                    hovertemplate=(
                        "X=%{x:.2f}<br>"
                        "Y=%{y:.2f}<br>"
                        "Amostras=%{z}<extra></extra>"
                    ),
                )
            )

            fig_heat.update_layout(
                title=f"Ocupação espacial — {title}",
                xaxis_title="X",
                yaxis_title="Y",
                height=500,
                margin=dict(l=20, r=20, t=55, b=20),
            )

            fig_heat.update_yaxes(
                scaleanchor="x",
                scaleratio=1,
            )

            st.plotly_chart(
                fig_heat,
                use_container_width=True
            )

# Tabela resumo
entropy_summary = pd.DataFrame({
    "Região": ["Campo total", "Claro", "Escuro"],
    "Entropia espacial": [
        Hsp_total, Hsp_clear, Hsp_dark
    ],
    "Entropia espacial normalizada": [
        Hspn_total, Hspn_clear, Hspn_dark
    ],
    "Entropia angular": [
        Hang_total, Hang_clear, Hang_dark
    ],
    "Entropia angular normalizada": [
        Hangn_total, Hangn_clear, Hangn_dark
    ],
})

st.dataframe(
    entropy_summary.round(4),
    use_container_width=True,
    hide_index=True,
)

st.download_button(
    "Baixar resumo de entropia em CSV",
    data=entropy_summary.to_csv(index=False).encode("utf-8"),
    file_name="resumo_entropia.csv",
    mime="text/csv",
)

st.caption(
    "As entropias normalizadas variam aproximadamente entre 0 e 1. "
    "A comparação entre animais deve ser feita usando o mesmo tamanho de célula "
    "e a mesma largura dos setores angulares."
)

st.subheader('Vetores na origem e elipses direcionais')
fige=go.Figure()
for name,dsub in [('Claro',clear),('Escuro',dark)]:
    if dsub.empty: continue
    fige.add_trace(go.Scatter(x=dsub['dx'],y=dsub['dy'],mode='markers',name=f'Vetores — {name}',marker=dict(size=4,opacity=.35)))
    ell=covariance_ellipse(dsub['dx'],dsub['dy'])
    if ell:
        fige.add_trace(go.Scatter(x=ell['x'],y=ell['y'],mode='lines',name=f'Elipse 95% — {name}',line=dict(width=3)))
fige.add_hline(y=0,line_width=1,line_dash='dot'); fige.add_vline(x=0,line_width=1,line_dash='dot')
fige.update_layout(xaxis_title='ΔX',yaxis_title='ΔY',height=600,legend=dict(orientation='h'))
fige.update_yaxes(scaleanchor='x',scaleratio=1)
st.plotly_chart(fige,use_container_width=True)

st.subheader('Distância percorrida por segundo')
tmp=seg.copy(); tmp['segundo']=np.floor(tmp['t_final']).astype(int)
ps=tmp.groupby(['segundo','lado'],as_index=False)['tamanho_vetor'].sum()
figs=go.Figure()
for name in ['Claro','Escuro']:
    q=ps[ps['lado']==name]
    if not q.empty:
        figs.add_trace(go.Scatter(x=q['segundo'],y=q['tamanho_vetor'],mode='lines',name=name))
figs.update_layout(xaxis_title='Tempo (s)',yaxis_title='Distância no segundo',height=420)
st.plotly_chart(figs,use_container_width=True)


st.subheader("Eventos de transição")

if len(transitions["crossing_times"]):
    transition_table = pd.DataFrame({
        "ordem": np.arange(1, len(transitions["crossing_times"]) + 1),
        "tempo_cruzamento": transitions["crossing_times"],
        "direcao": transitions["crossing_directions"],
        "velocidade_transicao": transitions["crossing_speeds"],
    })

    transition_table["intervalo_desde_transicao_anterior"] = np.nan
    if len(transition_table) > 1:
        transition_table.loc[
            transition_table.index[1:],
            "intervalo_desde_transicao_anterior"
        ] = np.diff(transitions["crossing_times"])

    st.dataframe(
        transition_table.round(4),
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "Baixar eventos de transição em CSV",
        data=transition_table.to_csv(index=False).encode("utf-8"),
        file_name="eventos_transicao_claro_escuro.csv",
        mime="text/csv",
    )
else:
    st.info("Nenhuma transição completa entre claro e escuro foi detectada no trecho selecionado.")

st.subheader('Exportação')
st.download_button('Baixar segmentos vetoriais em CSV',data=seg.to_csv(index=False).encode('utf-8'),file_name='segmentos_vetoriais_claro_escuro.csv',mime='text/csv')
st.download_button('Baixar resumo em CSV',data=comp.to_csv(index=False).encode('utf-8'),file_name='resumo_metricas_claro_escuro.csv',mime='text/csv')

with st.expander('Critérios de cálculo'):
    st.markdown('''
**Normalização espacial:** Xnormalizado = X − Xinicial. Assim, X>0 é escuro e X<0 é claro.

**Vetores:** ΔX = X(t+1)−X(t), ΔY = Y(t+1)−Y(t), tamanho = √(ΔX²+ΔY²), orientação = atan2(ΔY,ΔX), velocidade = tamanho/Δt.

**Cruzamentos de X=0:** cada deslocamento que atravessa a fronteira é dividido exatamente na interseção por interpolação linear. Tempo e distância são repartidos entre claro e escuro.

**Elipse direcional:** os endpoints (ΔX,ΔY) são colocados numa origem comum e é calculada uma elipse de covariância de 95%. O índice de direcionalidade é semieixo maior / semieixo menor.
''')

with st.expander('Outras variáveis úteis'):
    st.markdown('''
- percentual do tempo em cada lado;
- número de cruzamentos;
- velocidade média e DP por lado;
- orientação média circular;
- semieixos e orientação da elipse;
- índice de direcionalidade por lado.

Outras extensões possíveis: latência para primeira entrada, número de entradas, duração média das visitas, distância por visita, tempo imóvel por lado, índice de preferência claro/escuro, aceleração, tortuosidade e entropia espacial.
''')
