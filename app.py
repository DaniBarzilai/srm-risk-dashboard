import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# --- 1. CONFIGURAÇÃO E ESTILO ---
st.set_page_config(page_title="Monitor de Risco SRM", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: #FAFAFA; }
    div[data-testid="stMetricValue"] { font-size: 24px; color: #FAFAFA; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. CARREGAMENTO DE DADOS PRINCIPAIS ---
@st.cache_data
def carregar_dados():
    import zipfile
    try:
        try:
            # Tentativa 1: Trata como um ZIP real e desvia do fantasma do Mac
            zf = zipfile.ZipFile("dados_dashboard_validado.csv.zip")
            df = pd.read_csv(zf.open("dados_dashboard_validado.csv"))
        except zipfile.BadZipFile:
            # Tentativa 2: Leitura direta caso o ficheiro não esteja comprimido
            df = pd.read_csv("dados_dashboard_validado.csv.zip")
            
        df['Score'] = (100 * (1 - df['risco_predito'])).round(0)
        return df
    except Exception as e:
        st.error(f"Erro definitivo de leitura: {e}")
        return None

df = carregar_dados()

if df is None:
    st.stop()

# --- 3. NAVEGAÇÃO (SIDEBAR) ---
st.sidebar.title("🚀 SRM Ventures")
pagina = st.sidebar.selectbox("Navegação", 
    ["🏠 Home - Panorama", "📊 Detalhe por Fintech", "🔍 Por que este Score? (SHAP)", "⚠️ Alertas de Risco"])

# --- 4. LÓGICA DAS PÁGINAS ---

# PÁGINA 1: HOME
if pagina == "🏠 Home - Panorama":
    st.title("🎯 Panorama Geral da Carteira")
    
    st.subheader("Métricas de Negócio")
    col1, col2, col3 = st.columns(3)
    col1.metric("Score Médio SRM", f"{df['Score'].mean():.0f} / 100")
    col2.metric("Perda Esperada Total", f"R$ {df['Perda_Esperada_R$'].sum():,.2f}")
    col3.metric("Exposição Total", f"R$ {df['Valor'].sum():,.2f}")

    st.subheader("Validação Técnica do Modelo (XGBoost)")
    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("AUC-ROC", "0.74", "Boa separação de risco")
    col_m2.metric("Estatística KS", "0.34", "Boa distinção")
    col_m3.metric("Acurácia Global", "67%", "Taxa de acerto")

    st.markdown("---")
    
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Distribuição de Score")
        fig = px.histogram(df, x="Score", nbins=20, color_discrete_sequence=['#00CC96'], template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.subheader("Volume por Fintech")
        df_v = df.groupby('Fintech')['Valor'].sum().reset_index()
        fig_v = px.bar(df_v, x='Fintech', y='Valor', template="plotly_dark")
        st.plotly_chart(fig_v, use_container_width=True)

# PÁGINA 2: DETALHE POR FINTECH
elif pagina == "📊 Detalhe por Fintech":
    st.title("🔎 Análise Individualizada")
    st.sidebar.markdown("---")
    st.sidebar.subheader("Filtros da Fintech")
    fintech_sel = st.sidebar.selectbox("Selecione a Fintech:", df['Fintech'].unique())
    score_range = st.sidebar.slider("Filtrar por Faixa de Score:", 0, 100, (0, 100))

    df_f = df[(df['Fintech'] == fintech_sel) & (df['Score'] >= score_range[0]) & (df['Score'] <= score_range[1])]
    
    m1, m2, m3 = st.columns(3)
    score_atual = df_f['Score'].mean() if not df_f.empty else 0
    m1.metric(f"Score Médio - {fintech_sel}", f"{score_atual:.0f}")
    m2.metric("Total Operações", len(df_f))
    m3.metric("Valor em Risco", f"R$ {df_f['Perda_Esperada_R$'].sum():,.2f}")

    st.markdown("---")
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        st.subheader("Composição por Faixa de Risco")
        df_pizza = df_f['Faixa_Risco'].value_counts().reset_index()
        fig_pizza = px.pie(df_pizza, values='count', names='Faixa_Risco', hole=0.5, template="plotly_dark",
                           color='Faixa_Risco', color_discrete_map={'Baixo':'#00CC96', 'Médio':'#FFA15A', 'Alto':'#EF553B'})
        st.plotly_chart(fig_pizza, use_container_width=True)
    with col_g2:
        st.subheader("Relação Valor x Score")
        fig_scatter = px.scatter(df_f, x="Score", y="Valor", color="Faixa_Risco", template="plotly_dark")
        st.plotly_chart(fig_scatter, use_container_width=True)

    # ---GRÁFICO DE SHAP LOCAL ---
    st.markdown("---")
    st.subheader(f"🧠 Raio-X de Risco: {fintech_sel}")
    try:
        df_shap_local = pd.read_csv("shap_local_fintechs.csv")
        # Isolar os dados da fintech escolhida no menu
        df_shap_f = df_shap_local[df_shap_local['Fintech'] == fintech_sel].sort_values(by='Impacto', ascending=True)

        if not df_shap_f.empty:
            fig_shap_local = px.bar(df_shap_f, x='Impacto', y='Variável', orientation='h',
                                    color_discrete_sequence=['#FFA15A'], template="plotly_dark")
            st.plotly_chart(fig_shap_local, use_container_width=True)
            st.info("💡 **Dica:** As barras maiores mostram exatamente quais os fatores específicos que puxam o score desta fintech para baixo (ou para cima).")
        else:
            st.warning("Sem dados de impacto detalhado para esta Fintech.")
    except FileNotFoundError:
        st.warning("⚠️ O ficheiro 'shap_local_fintechs.csv' não foi encontrado.")

# PÁGINA 3: SHAP
elif pagina == "🔍 Por que este Score? (SHAP)":
    st.title("🧠 Inteligência do Modelo (SHAP Global)")
    st.markdown("Entenda quais variáveis reais mais impactam o Score final de crédito do seu modelo XGBoost de forma geral na SRM.")
    
    try:
        df_shap = pd.read_csv("shap_global_srm.csv")
        df_shap_top = df_shap.head(10).sort_values(by='Impacto', ascending=True)
        
        fig_shap = px.bar(df_shap_top, x='Impacto', y='Variável', orientation='h',
                          title="Top 10 Variáveis de Maior Impacto no Risco",
                          color_discrete_sequence=['#636EFA'], template="plotly_dark")
        st.plotly_chart(fig_shap, use_container_width=True)
    except FileNotFoundError:
        st.warning("⚠️ O ficheiro 'shap_global_srm.csv' não foi encontrado.")

# PÁGINA 4: ALERTAS
elif pagina == "⚠️ Alertas de Risco":
    st.title("⚠️ Painel de Monitoramento Crítico")
    df_a = df[df['Faixa_Risco'] == 'Alto'].sort_values(by='Score', ascending=True)
    
    if df_a.empty:
        st.success("Tudo certo! Nenhuma operação de alto risco detetada no momento.")
    else:
        st.error(f"Existem {len(df_a)} operações em nível CRÍTICO.")
        st.dataframe(df_a[['Fintech', 'Valor', 'Score', 'prazo_pagamento_dias']], use_container_width=True)
