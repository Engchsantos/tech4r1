import pandas as pd
import streamlit as st
from prophet import Prophet


#Importando e ajustando a base de dados

df = pd.read_html('http://www.ipeadata.gov.br/ExibeSerie.aspx?module=m&serid=1650971490&oper=view',
                 encoding ="utf-8",
                 thousands='.',
                 decimal=',',
                 header = 0
                 
)
df = df[2]
df['Data'] = pd.to_datetime(df['Data'], dayfirst=True)
df.rename(columns={'Preço - petróleo bruto - Brent (FOB)': 'y'}, inplace=True)
df.rename(columns={'Data': 'ds'}, inplace=True)
df.reset_index(drop=True, inplace=True)

preco_atual = df['y'][0]
preco_anterior = df['y'][1]


#Criando Modelo Prophet
train = df.iloc[:-365]
test = df.iloc[-365:]

m = Prophet()
m.fit(df)
dias = m.make_future_dataframe(periods=1)
previsao = m.predict(dias)

prev_amanha = previsao['yhat'][-1:]

preco_amanha = prev_amanha.values

# Configurando a página no Streamlit

st.set_page_config(
    page_title= 'TECH CHALLENGE4',
    layout= 'wide'
)

st.header("**PREÇO DO PETRÓLEO BRENT - US$**")

st.line_chart(data=df, x='ds', y='y')

st.header("**PREÇOS**")
col1, col2, col3 = st.columns([1,1,1])

with col1:
    st.write(f"**Preço Anterior:** {preco_anterior}")
with col2:
    st.write(f"**Preço Atual:** {preco_atual}")
with col3:
    st.write(f"**Previsão Para Amanhã:** {preco_amanha} ")
