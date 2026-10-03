
import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(
    page_title="Análise de Trajetória 2D",
    page_icon="📍",
    layout="wide",
)

st.title("Análise de Trajetória 2D")
st.caption(
    "Importe um CSV com tempo, posição X e posição Y. "
    "Use o controle de tempo para acompanhar a trajetória sendo formada e recalcular as métricas."
)

# -----------------------------
# Funções auxiliares
# -----------------------------
def read_csv_flexible(uploaded_file):
    raw = uploaded_file.getvalue()

    # Tenta algumas codificações comuns
    text = None
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            pass

    if text is None:
        raise ValueError("Não foi possível decodificar o arquivo CSV.")

    # Inferência de separador
    try:
        df = pd.read_csv(io.StringIO(text), sep=None, engine="python")
    except Exception:
        for sep in (",", ";", "\t"):
            try:
                df = pd.read_csv(io.StringIO(text), sep=sep)
                if df.shape[1] >= 3:
                    break
            except Exception:
                df = None
        if df is None:
            raise ValueError("Não foi possível interpretar o CSV.")

    if df.shape[1] < 3:
        raise ValueError("O arquivo precisa ter pelo menos 3 colunas: tempo, X e Y.")

    return df


def to_numeric_flexible(series):
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    s = series.astype(str).str.strip()

    # Se houver vírgula decimal e não houver ponto, converte vírgula para ponto.
    comma_fraction = s.str.contains(",", regex=False).mean()
    dot_fraction = s.str.contains(".", regex=False).mean()
    if comma_fraction > 0.3 and dot_fraction < 0.3:
        s = s.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)

    return pd.to_numeric(s, errors="coerce")


def prepare_data(df):
    # Por padrão, usa as 3 primeiras colunas independentemente do cabeçalho.
    out = pd.DataFrame({
        "tempo": to_numeric_flexible(df.iloc[:, 0]),
        "x": to_numeric_flexible(df.iloc[:, 1]),
        "y": to_numeric_flexible(df.iloc[:, 2]),
    })

    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    out = out.sort_values("tempo").drop_duplicates(subset="tempo", keep="first").reset_index(drop=True)

    if len(out) < 2:
        raise ValueError("São necessárias pelo menos duas amostras válidas.")

    # Faz o tempo começar em zero para facilitar a visualização.
    out["tempo_original"] = out["tempo"]
    out["tempo"] = out["tempo"] - out["tempo"].iloc[0]
    return out


def vector_table(df):
    t = df["tempo"].to_numpy(float)
    x = df["x"].to_numpy(float)
    y = df["y"].to_numpy(float)

    dt = np.diff(t)
    dx = np.diff(x)
    dy = np.diff(y)
    length = np.hypot(dx, dy)

    # Orientação em graus no intervalo [0, 360)
    angle_deg = (np.degrees(np.arctan2(dy, dx)) + 360.0) % 360.0

    valid_speed = dt > 0
    speed = np.full_like(length, np.nan, dtype=float)
    speed[valid_speed] = length[valid_speed] / dt[valid_speed]

    return pd.DataFrame({
        "t_inicial": t[:-1],
        "t_final": t[1:],
        "dt": dt,
        "x_inicial": x[:-1],
        "y_inicial": y[:-1],
        "x_final": x[1:],
        "y_final": y[1:],
        "dx": dx,
        "dy": dy,
        "tamanho_vetor": length,
        "orientacao_graus": angle_deg,
        "velocidade_intervalo": speed,
    })


def covariance_ellipse(dx, dy, confidence_95=True):
    points = np.column_stack([dx, dy])
    finite = np.all(np.isfinite(points), axis=1)
    points = points[finite]

    if len(points) < 3:
        return None

    cov = np.cov(points, rowvar=False, ddof=1)
    if not np.all(np.isfinite(cov)):
        return None

    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = np.maximum(eigvals[order], 0.0)
    eigvecs = eigvecs[:, order]

    # Para uma elipse gaussiana bivariada de 95%, chi2(df=2, 0.95)=5.991
    scale = np.sqrt(5.991) if confidence_95 else 1.0

    major = scale * np.sqrt(eigvals[0])
    minor = scale * np.sqrt(eigvals[1])

    # Orientação do eixo maior
    vx, vy = eigvecs[:, 0]
    ellipse_angle = (np.degrees(np.arctan2(vy, vx)) + 180.0) % 180.0

    theta = np.linspace(0, 2*np.pi, 361)
    unit = np.vstack([np.cos(theta), np.sin(theta)])
    transform = eigvecs @ np.diag([major, minor])
    ellipse = transform @ unit

    directionality = np.inf if minor == 0 else major / minor
    inverse_index = 0.0 if major == 0 else minor / major

    return {
        "major": major,
        "minor": minor,
        "directionality": directionality,
        "inverse_index": inverse_index,
        "angle": ellipse_angle,
        "ellipse_x": ellipse[0],
        "ellipse_y": ellipse[1],
        "cov": cov,
    }


def circular_mean_deg(angles_deg, weights=None):
    a = np.deg2rad(np.asarray(angles_deg, dtype=float))
    if len(a) == 0:
        return np.nan
    if weights is None:
        weights = np.ones_like(a)
    weights = np.asarray(weights, dtype=float)
    ok = np.isfinite(a) & np.isfinite(weights)
    if not np.any(ok):
        return np.nan
    s = np.sum(weights[ok] * np.sin(a[ok]))
    c = np.sum(weights[ok] * np.cos(a[ok]))
    return (np.degrees(np.arctan2(s, c)) + 360.0) % 360.0


def fmt(x, digits=3):
    if x is None or not np.isfinite(x):
        return "—"
    return f"{x:.{digits}f}"


# -----------------------------
# Upload
# -----------------------------
uploaded = st.file_uploader("Selecione o arquivo CSV", type=["csv"])

if uploaded is None:
    st.info("Carregue um arquivo como `tempo_posicao_xy.csv` para iniciar.")
    st.stop()

try:
    raw_df = read_csv_flexible(uploaded)
    df = prepare_data(raw_df)
except Exception as exc:
    st.error(f"Erro ao ler o arquivo: {exc}")
    st.stop()

total_time = float(df["tempo"].iloc[-1])

# -----------------------------
# Controle temporal
# -----------------------------
st.subheader("Controle da trajetória")

col_slider, col_play = st.columns([5, 1])

with col_slider:
    selected_time = st.slider(
        "Tempo acumulado (s)",
        min_value=0.0,
        max_value=total_time,
        value=total_time,
        step=max(total_time / 1000.0, 0.01),
        format="%.2f s",
    )

with col_play:
    st.metric("Duração total", f"{total_time:.2f} s")

current = df[df["tempo"] <= selected_time].copy()

# Garante pelo menos o primeiro ponto
if current.empty:
    current = df.iloc[[0]].copy()

# -----------------------------
# Trajetória
# -----------------------------
st.subheader("Trajetória espacial")

fig_traj = go.Figure()

fig_traj.add_trace(go.Scatter(
    x=current["x"],
    y=current["y"],
    mode="lines",
    name="Trajetória",
    line=dict(width=2),
    hovertemplate="X=%{x:.3f}<br>Y=%{y:.3f}<extra></extra>",
))

fig_traj.add_trace(go.Scatter(
    x=[current["x"].iloc[0]],
    y=[current["y"].iloc[0]],
    mode="markers",
    name="Início",
    marker=dict(size=10, symbol="circle"),
))

fig_traj.add_trace(go.Scatter(
    x=[current["x"].iloc[-1]],
    y=[current["y"].iloc[-1]],
    mode="markers",
    name="Posição atual",
    marker=dict(size=11, symbol="diamond"),
))

fig_traj.update_layout(
    xaxis_title="X",
    yaxis_title="Y",
    height=560,
    margin=dict(l=20, r=20, t=30, b=20),
    legend=dict(orientation="h"),
)

# Mantém escalas espaciais equivalentes
fig_traj.update_yaxes(scaleanchor="x", scaleratio=1)

st.plotly_chart(fig_traj, use_container_width=True)

if len(current) < 2:
    st.warning("Aumente o tempo para incluir pelo menos dois pontos e gerar vetores.")
    st.stop()

vectors = vector_table(current)

# -----------------------------
# Métricas
# -----------------------------
elapsed = float(current["tempo"].iloc[-1] - current["tempo"].iloc[0])
distance_total = float(vectors["tamanho_vetor"].sum())
distance_per_second = distance_total / elapsed if elapsed > 0 else np.nan
mean_vector_length = float(vectors["tamanho_vetor"].mean())
sd_vector_length = float(vectors["tamanho_vetor"].std(ddof=1)) if len(vectors) > 1 else np.nan
mean_orientation = circular_mean_deg(
    vectors["orientacao_graus"].to_numpy(),
    weights=vectors["tamanho_vetor"].to_numpy(),
)

ellipse = covariance_ellipse(
    vectors["dx"].to_numpy(),
    vectors["dy"].to_numpy(),
    confidence_95=True,
)

st.subheader("Métricas espaciais")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Distância total", fmt(distance_total))
m2.metric("Distância / segundo", fmt(distance_per_second))
m3.metric("Tamanho médio do vetor", fmt(mean_vector_length))
m4.metric("DP do tamanho do vetor", fmt(sd_vector_length))

m5, m6, m7, m8 = st.columns(4)
m5.metric("Nº de vetores", f"{len(vectors)}")
m6.metric("Orientação média*", f"{fmt(mean_orientation, 1)}°")

if ellipse is not None:
    m7.metric("Semieixo maior (95%)", fmt(ellipse["major"]))
    m8.metric("Semieixo menor (95%)", fmt(ellipse["minor"]))

    d1, d2, d3 = st.columns(3)
    d1.metric("Índice de direcionalidade", fmt(ellipse["directionality"]))
    d2.metric("Índice inverso (menor/maior)", fmt(ellipse["inverse_index"]))
    d3.metric("Orientação da elipse", f"{fmt(ellipse['angle'], 1)}°")
else:
    st.warning("Ainda não há vetores suficientes para estimar a elipse.")

st.caption(
    "* A orientação média é uma média circular ponderada pelo tamanho dos vetores. "
    "O índice de direcionalidade principal é semieixo maior / semieixo menor: "
    "valores próximos de 1 indicam distribuição mais isotrópica; valores maiores indicam maior anisotropia direcional."
)

# -----------------------------
# Vetores na origem + elipse
# -----------------------------
st.subheader("Vetores espaciais transladados para a origem")

fig_vec = go.Figure()

dx = vectors["dx"].to_numpy()
dy = vectors["dy"].to_numpy()

# Nuvem dos endpoints
fig_vec.add_trace(go.Scatter(
    x=dx,
    y=dy,
    mode="markers",
    name="Endpoints dos vetores",
    marker=dict(size=5, opacity=0.5),
    hovertemplate="ΔX=%{x:.3f}<br>ΔY=%{y:.3f}<extra></extra>",
))

# Desenha uma amostra das setas para não sobrecarregar o gráfico
max_arrows = 250
if len(dx) <= max_arrows:
    idx = np.arange(len(dx))
else:
    idx = np.linspace(0, len(dx)-1, max_arrows).astype(int)

for i in idx:
    fig_vec.add_shape(
        type="line",
        x0=0, y0=0,
        x1=float(dx[i]), y1=float(dy[i]),
        line=dict(width=1),
        opacity=0.15,
    )

if ellipse is not None:
    fig_vec.add_trace(go.Scatter(
        x=ellipse["ellipse_x"],
        y=ellipse["ellipse_y"],
        mode="lines",
        name="Elipse de covariância 95%",
        line=dict(width=3),
        hoverinfo="skip",
    ))

    # Eixos principais
    phi = np.deg2rad(ellipse["angle"])
    major_dx = ellipse["major"] * np.cos(phi)
    major_dy = ellipse["major"] * np.sin(phi)

    minor_phi = phi + np.pi/2
    minor_dx = ellipse["minor"] * np.cos(minor_phi)
    minor_dy = ellipse["minor"] * np.sin(minor_phi)

    fig_vec.add_trace(go.Scatter(
        x=[-major_dx, major_dx],
        y=[-major_dy, major_dy],
        mode="lines",
        name="Eixo maior",
        line=dict(width=3, dash="dash"),
    ))

    fig_vec.add_trace(go.Scatter(
        x=[-minor_dx, minor_dx],
        y=[-minor_dy, minor_dy],
        mode="lines",
        name="Eixo menor",
        line=dict(width=3, dash="dot"),
    ))

fig_vec.update_layout(
    xaxis_title="ΔX",
    yaxis_title="ΔY",
    height=600,
    margin=dict(l=20, r=20, t=30, b=20),
    legend=dict(orientation="h"),
)
fig_vec.update_yaxes(scaleanchor="x", scaleratio=1)

st.plotly_chart(fig_vec, use_container_width=True)

# -----------------------------
# Orientações
# -----------------------------
st.subheader("Distribuição das orientações dos vetores")

bins = np.arange(0, 361, 15)
hist, edges = np.histogram(vectors["orientacao_graus"], bins=bins)
centers = (edges[:-1] + edges[1:]) / 2

fig_ang = go.Figure(go.Bar(
    x=centers,
    y=hist,
    width=13,
    name="Vetores",
))

fig_ang.update_layout(
    xaxis=dict(
        title="Orientação (graus)",
        tickmode="array",
        tickvals=np.arange(0, 361, 45),
        range=[0, 360],
    ),
    yaxis_title="Número de vetores",
    height=400,
    margin=dict(l=20, r=20, t=30, b=20),
)

st.plotly_chart(fig_ang, use_container_width=True)

# -----------------------------
# Distância percorrida por segundo
# -----------------------------
st.subheader("Distância percorrida por segundo")

# Atribui cada deslocamento ao segundo de seu tempo final
second_index = np.floor(vectors["t_final"]).astype(int)
per_second = (
    vectors.assign(segundo=second_index)
    .groupby("segundo", as_index=False)["tamanho_vetor"]
    .sum()
    .rename(columns={"tamanho_vetor": "distancia"})
)

fig_sec = go.Figure(go.Scatter(
    x=per_second["segundo"],
    y=per_second["distancia"],
    mode="lines",
    name="Distância por segundo",
))

fig_sec.update_layout(
    xaxis_title="Tempo (s)",
    yaxis_title="Distância percorrida no segundo",
    height=400,
    margin=dict(l=20, r=20, t=30, b=20),
)

st.plotly_chart(fig_sec, use_container_width=True)

# -----------------------------
# Dados e exportação
# -----------------------------
with st.expander("Ver tabela dos vetores"):
    st.dataframe(vectors, use_container_width=True, height=350)

csv_vectors = vectors.to_csv(index=False).encode("utf-8")
st.download_button(
    "Baixar tabela de vetores em CSV",
    data=csv_vectors,
    file_name="vetores_espaciais.csv",
    mime="text/csv",
)

with st.expander("Como os cálculos são feitos"):
    st.markdown(
        """
        Para cada par de coordenadas consecutivas:

        - **ΔX = X(t+1) − X(t)**
        - **ΔY = Y(t+1) − Y(t)**
        - **Tamanho do vetor = √(ΔX² + ΔY²)**
        - **Orientação = atan2(ΔY, ΔX)**, convertida para graus entre 0° e 360°
        - **Distância total** = soma dos tamanhos de todos os vetores
        - **Distância por segundo** = distância total / duração analisada
        - **Tamanho médio do vetor** = média dos comprimentos
        - **DP do tamanho do vetor** = desvio-padrão amostral dos comprimentos

        Para a análise direcional, cada vetor é transladado para a origem, de modo que seu endpoint
        fique em **(ΔX, ΔY)**. A matriz de covariância desses endpoints é diagonalizada. Os autovetores
        definem as direções principais da elipse e as raízes dos autovalores definem seus semieixos.
        O gráfico usa uma elipse de covariância de **95%**.

        O **índice de direcionalidade = semieixo maior / semieixo menor**.
        Assim, **1** representa uma distribuição aproximadamente isotrópica, enquanto valores progressivamente
        maiores indicam maior anisotropia espacial.
        """
    )
