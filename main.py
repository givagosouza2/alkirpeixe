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
        mat = loadmat(io.BytesIO(uploaded_file.getvalue()), squeeze_me=True, struct_as_record=False)
    except NotImplementedError:
        raise ValueError('Arquivo MATLAB v7.3/HDF5 não suportado por scipy.io.loadmat nesta versão.')
    except Exception as exc:
        raise ValueError(f'Não foi possível abrir o .mat: {exc}')

    if 'e' in mat:
        e = mat['e']
        try:
            return pd.DataFrame({
                'tempo': np.asarray(e.t, dtype=float).ravel(),
                'x': np.asarray(e.posicao.x, dtype=float).ravel(),
                'y': np.asarray(e.posicao.y, dtype=float).ravel(),
            })
        except Exception:
            pass

    keys = {k.lower(): k for k in mat.keys() if not k.startswith('__')}
    tkey = next((keys[k] for k in keys if k in ('t','tempo','time')), None)
    xkey = next((keys[k] for k in keys if k in ('x','posicao_x','position_x')), None)
    ykey = next((keys[k] for k in keys if k in ('y','posicao_y','position_y')), None)
    if tkey and xkey and ykey:
        return pd.DataFrame({
            'tempo': np.asarray(mat[tkey], dtype=float).ravel(),
            'x': np.asarray(mat[xkey], dtype=float).ravel(),
            'y': np.asarray(mat[ykey], dtype=float).ravel(),
        })
    raise ValueError('Não encontrei e.t, e.posicao.x e e.posicao.y no .mat.')


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
    if uploaded.name.lower().endswith('.mat'):
        raw=extract_from_mat(uploaded); ftype='MATLAB'
    else:
        raw=extract_from_csv(uploaded); ftype='CSV'
    df,x0orig=prepare_data(raw)
except Exception as exc:
    st.error(str(exc)); st.stop()

st.success(f'{ftype} carregado: {len(df)} amostras. X inicial original={x0orig:.4f}; novo X inicial=0.')

tmax=float(df['tempo'].iloc[-1])
selected=st.slider('Mostrar trajetória até',0.0,tmax,tmax,step=max(tmax/1500,0.01),format='%.2f s')
cur=df[df['tempo']<=selected].copy()
if len(cur)<2:
    st.warning('Aumente o tempo para incluir pelo menos duas coordenadas.'); st.stop()
seg=split_segments(cur)
if seg.empty:
    st.warning('Não foi possível formar segmentos válidos.'); st.stop()
clear=seg[seg['lado']=='Claro'].copy(); dark=seg[seg['lado']=='Escuro'].copy()
sall=summarize(seg,'Campo total'); sc=summarize(clear,'Claro'); sd=summarize(dark,'Escuro')
T=float(cur['tempo'].iloc[-1]-cur['tempo'].iloc[0]); D=float(seg['tamanho_vetor'].sum())
mean_speed=D/T if T>0 else np.nan
speed_sd=float(seg['velocidade'].std(ddof=1)) if len(seg)>1 else np.nan
cross=count_crossings(cur)

st.subheader('Trajetória')
fig=go.Figure()
fig.add_trace(go.Scatter(x=cur['x'],y=cur['y'],mode='lines',name='Trajetória',line=dict(width=2)))
fig.add_trace(go.Scatter(x=cur.loc[cur['x']<=0,'x'],y=cur.loc[cur['x']<=0,'y'],mode='markers',name='Claro',marker=dict(size=4,opacity=.4)))
fig.add_trace(go.Scatter(x=cur.loc[cur['x']>=0,'x'],y=cur.loc[cur['x']>=0,'y'],mode='markers',name='Escuro',marker=dict(size=4,opacity=.4)))
fig.add_vline(x=0,line_width=2,line_dash='dash',annotation_text='X = 0')
fig.add_trace(go.Scatter(x=[cur['x'].iloc[0]],y=[cur['y'].iloc[0]],mode='markers',name='Início',marker=dict(size=11)))
fig.add_trace(go.Scatter(x=[cur['x'].iloc[-1]],y=[cur['y'].iloc[-1]],mode='markers',name='Atual',marker=dict(size=11,symbol='diamond')))
fig.update_layout(xaxis_title='X normalizado',yaxis_title='Y',height=600,legend=dict(orientation='h'))
fig.update_yaxes(scaleanchor='x',scaleratio=1)
st.plotly_chart(fig,use_container_width=True)

st.subheader('Métricas gerais')
cols=st.columns(4)
cols[0].metric('Distância total',fmt(D)); cols[1].metric('Tempo total',f'{fmt(T,2)} s'); cols[2].metric('Velocidade média',fmt(mean_speed)); cols[3].metric('DP velocidade',fmt(speed_sd))
cols=st.columns(4)
cols[0].metric('Tempo claro',f"{fmt(sc['tempo_total'],2)} s"); cols[1].metric('Tempo escuro',f"{fmt(sd['tempo_total'],2)} s"); cols[2].metric('% tempo claro',f"{fmt(100*sc['tempo_total']/T if T>0 else np.nan,1)}%"); cols[3].metric('% tempo escuro',f"{fmt(100*sd['tempo_total']/T if T>0 else np.nan,1)}%")
cols=st.columns(3)
cols[0].metric('Distância claro',fmt(sc['distancia_total'])); cols[1].metric('Distância escuro',fmt(sd['distancia_total'])); cols[2].metric('Cruzamentos',str(cross))

comp=pd.DataFrame([sall,sc,sd])
comp=comp[['regiao','tempo_total','distancia_total','velocidade_media','velocidade_dp','vetor_medio','vetor_dp','orientacao_media','semieixo_maior','semieixo_menor','indice_direcionalidade','orientacao_elipse','n_segmentos']]
comp.columns=['Região','Tempo total','Distância total','Velocidade média','DP velocidade','Tamanho médio vetor','DP tamanho vetor','Orientação média (°)','Semieixo maior','Semieixo menor','Índice direcionalidade','Orientação elipse (°)','N segmentos']
st.subheader('Comparação entre regiões')
st.dataframe(comp.round(4),use_container_width=True,hide_index=True)

st.subheader('Velocidade ao longo do tempo')
figv=go.Figure()
for name,dsub in [('Claro',clear),('Escuro',dark)]:
    if not dsub.empty:
        figv.add_trace(go.Scatter(x=dsub['t_final'],y=dsub['velocidade'],mode='markers',name=name,marker=dict(size=4,opacity=.55)))
figv.update_layout(xaxis_title='Tempo (s)',yaxis_title='Velocidade',height=420)
st.plotly_chart(figv,use_container_width=True)

st.subheader('Distribuição dos ângulos')
bw=st.select_slider('Largura dos setores',options=[5,10,15,20,30,45],value=15,format_func=lambda x:f'{x}°')
bins=np.arange(0,360+bw,bw); centers=(bins[:-1]+bins[1:])/2
figa=go.Figure()
for name,dsub in [('Claro',clear),('Escuro',dark)]:
    if not dsub.empty:
        h,_=np.histogram(dsub['orientacao_graus'],bins=bins)
        figa.add_trace(go.Bar(x=centers,y=h,name=name,opacity=.75))
figa.update_layout(barmode='group',xaxis=dict(title='Orientação (°)',range=[0,360],tickmode='array',tickvals=np.arange(0,361,45)),yaxis_title='Número de vetores',height=420)
st.plotly_chart(figa,use_container_width=True)

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
