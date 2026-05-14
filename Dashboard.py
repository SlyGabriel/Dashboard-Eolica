import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry
import folium
import streamlit as st
import openmeteo_requests
from folium.plugins import HeatMap
import numpy as np
from streamlit_folium import st_folium
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
# Iniciacilização da API, retornando cache e error
cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)

#Parâmetros para o cálculo do potencial energético das turbinas
den = 1.225 # densidade do ar
raio = 60  # metros (tubina pequena)
area = np.pi*raio**2
eficiencia = 0.593

st.set_page_config(
    page_title="Dashboard Eólica IA",
    layout="wide"
)

st.title("Dashboard Inteligente de Energia Eólica")
st.markdown(
    "Análise de potencial eólico, previsões por IA e monitoramento energético do Brasil"
)
st.markdown("---")

def calcular_potêncial(velocidade):
	return 0.5*den*area*(velocidade**3) * eficiencia
estados_norte = [
	{"nome": "Amazonas", "lat": -3.10, "lon": -60.02,"regiao": "Norte"},
    {"nome": "Pará", "lat": -1.45, "lon": -48.50,"regiao": "Norte"},
    {"nome": "Acre", "lat": -9.97, "lon": -67.81,"regiao": "Norte"},
    {"nome": "Rondônia", "lat": -8.76, "lon": -63.90,"regiao": "Norte"},
    {"nome": "Roraima", "lat": 2.82, "lon": -60.67,"regiao": "Norte"},
    {"nome": "Amapá", "lat": 0.03, "lon": -51.05,"regiao": "Norte"},
    {"nome": "Tocantins", "lat": -10.25, "lon": -48.25,"regiao": "Norte"}
]

estados_sul = [
	{"nome": "Curitiba", "lat": -25.42,"lon": -49.22,"regiao": "Sul"},
    {"nome": "Florianópolis", "lat": -27.57,"lon": -48.53,"regiao": "Sul"},
    {"nome": "Porto Alegre", "lat": -29.96, "lon": -51.27,"regiao": "Sul"},
   
]

estados_sudeste =[
	{"nome": "São Paulo", "lat": -23.59,"lon": -46.59,"regiao": "Sudeste"},
    {"nome": "Rio de Janeiro", "lat": -22.87,"lon": -43.19,"regiao": "Sudeste"},
    {"nome": "Espírito Santo", "lat": -20.33,"lon": -40.34,"regiao": "Sudeste"},
    {"nome": "Minas Gerais", "lat": -19.97,"lon": -43.94,"regiao": "Sudeste"},

]

estados_nordeste = [
	{"nome": "Alagoas", "lat": -9.65,"lon": -35.73,"regiao": "Nordeste"},
    {"nome": "Bahia", "lat": -12.94,"lon": -38.49,"regiao": "Nordeste"},
    {"nome": "Ceará", "lat": -3.71, "lon": -38.26,"regiao": "Nordeste"},
    {"nome": "Maranhão", "lat": -2.55, "lon": -44.34,"regiao": "Nordeste"},
    {"nome": "Paraíba", "lat": -7.10,"lon":-34.90,"regiao": "Nordeste"},
    {"nome": "Pernambuco", "lat": -8.08,"lon": -34.87,"regiao": "Nordeste"},
    {"nome": "Piauí", "lat": -5.07,"lon": -42.74,"regiao": "Nordeste"},
	{"nome": "Rio Grande do Norte", "lat": -5.80,"lon": -35.19,"regiao": "Nordeste"},
    {"nome": "Sergipe", "lat": -10.92,"lon": -37.04,"regiao": "Nordeste"}
]

estados_centro_oeste = [
	{"nome": "Mato Grosso", "lat": -15.53,"lon": -55.57,"regiao": "Centro Oeste"},
    {"nome": "Goiás", "lat": -16.67,"lon": -49.29,"regiao": "Centro Oeste"},
    {"nome": "Mato Grosso do Sul", "lat": -20.44,"lon": -54.046,"regiao": "Centro Oeste"},

]
def pegar_media(lat, lon, nome_estado, regiao):

	url = "https://api.open-meteo.com/v1/forecast"
	params = {
	"latitude": lat,
	"longitude": lon,
	"hourly": ["apparent_temperature", "wind_speed_120m", "temperature_120m"],
	}
	responses = openmeteo.weather_api(url, params=params)
	response = responses[0]
	hourly = response.Hourly()

	df = pd.DataFrame({
		"temp_ap" : hourly.Variables(0).ValuesAsNumpy(),
		"vento" : hourly.Variables(1).ValuesAsNumpy(),
		"temp_120m" : hourly.Variables(2).ValuesAsNumpy()
	})
	
	df["tempo_num"] = np.arange(len(df))

	X = df[["tempo_num"]]
	y = df["vento"]

	modelo = LinearRegression()
	modelo.fit(X, y)

	futuro_7dias = np.arange(
    	len(df),
    	len(df) + 168
	).reshape(-1, 1)

	previsao_7dias = modelo.predict(futuro_7dias)

	previsao_media7dias = previsao_7dias.mean()

	return{
		"estado" : nome_estado,
		"lat" : lat,
		"lon" : lon,
		"media_vento" : df["vento"].mean(),
		"media_temp_ap": df["temp_ap"].mean(),
		"media_temp_120": df["temp_120m"].mean(),
		"regiao": regiao,
		"vento_7dias": previsao_media7dias,
		"previsao_array": previsao_7dias.tolist(),
	}

class Regioes:

	def __init__(self, estados):
		self.estados = estados

	def gerar_dataframe(self):
		dados = []
	
		for estado in self.estados:
			dados.append(
			pegar_media(estado["lat"], estado["lon"], estado["nome"], estado["regiao"])
			)
		return pd.DataFrame(dados)
@st.cache_data		
def gerar_df_brasil():
	return pd.concat([
	Regioes(estados_norte).gerar_dataframe(),
    Regioes(estados_sul).gerar_dataframe(),
    Regioes(estados_sudeste).gerar_dataframe(),
    Regioes(estados_nordeste).gerar_dataframe(),
    Regioes(estados_centro_oeste).gerar_dataframe()
	])

def cor_vento(valor):
    if valor < 5:
        return "green"
    elif valor < 8:
        return "orange"
    else:
        return "red"
	

df_brasil = gerar_df_brasil()
df_brasil = gerar_df_brasil().reset_index(drop=True)
df_brasil["potencia_watts"] = df_brasil["media_vento"].apply(calcular_potêncial)
df_brasil["potencia_kw"] = df_brasil["potencia_watts"] / 1000

def classificar(p):
	if p < 500:
		return "Baixo"
	elif p < 1500:
		return "Medio"
	else:
		return "Alto"
	
df_brasil["classe_potencial"] = df_brasil["potencia_kw"].apply(classificar)
	
media_nacional = df_brasil["potencia_kw"].mean()

media_regioes = df_brasil.groupby(
	"regiao"
)["potencia_kw"].mean()

melhor_regiao = media_regioes.idxmax()

valor = media_regioes.max()

indice_maior = df_brasil["potencia_kw"].idxmax()

estado_maior = df_brasil.loc[
    indice_maior,
    "estado"
]

potencia_maior = float(
    df_brasil.loc[
        indice_maior,
        "potencia_kw"
    ]
)

ranking = df_brasil.sort_values(
	by="potencia_kw",
	ascending=False
).reset_index(drop=True)
ranking.index = ranking.index + 1



aba0, aba1, aba2, aba3 = st.tabs([
	"Análise por estado",
	"Mapa",
	"Análise IA",
	"Ranking engergético"
])

with aba0:
	regiao = st.selectbox(
    	"Escolha o estado",
    	df_brasil["estado"].unique(),
	)

	df_filtrado = df_brasil[df_brasil["estado"] == regiao]

	st.metric(
    	"Vento médio",
    	f"{df_filtrado['media_vento'].mean():.2f} m/s"
	)

	st.metric(
    	"Temperatura",
    	f"{df_filtrado['media_temp_ap'].mean():.2f} °C"
	)

	st.metric(
		"Potencial Eólico",
		f"{df_filtrado['potencia_kw'].mean():.2f} kw"
	)
legenda_html = """
<div style="
position: fixed;
bottom: 50px;
left: 50px;
width: 180px;
height: 170px;
background-color: black;
border:2px solid white;
z-index:9999;
font-size:14px;
padding: 10px;
">

<b>Legenda dos Ventos</b><br><br>

<i style="background:green;
width:10px;
height:10px;
float:left;
margin-right:8px;
opacity:0.7;
"></i>
Baixo (&lt; 5 m/s)<br><br>

<i style="background:orange;
width:10px;
height:10px;
float:left;
margin-right:8px;
opacity:0.7;
"></i>
Médio (5 - 8 m/s)<br><br>

<i style="background:red;
width:10px;
height:10px;
float:left;
margin-right:8px;
opacity:0.7;
"></i>
Alto (&gt; 8 m/s)

</div>
"""
mapa = folium.Map(
        location=[-14, -52],
        zoom_start=4
    )

with aba1:

    mapa = folium.Map(
        location=[-14, -52],
        zoom_start=4
    )

    tipo_mapa = st.selectbox(
        "Tipo de visualização",
        ["Círculos", "HeatMap", "Potencial Eólico"]
    )

    if tipo_mapa == "Círculos":

        mapa.get_root().html.add_child(
            folium.Element(legenda_html)
        )

        for _, row in df_brasil.iterrows():

            folium.CircleMarker(
                location=[row['lat'], row['lon']],
                radius=row['media_vento'],
                popup=f"""
                Estado: {row['estado']}<br>
                Vento: {row['media_vento']:.2f} m/s
                """,
                color=cor_vento(row['media_vento']),
                fill=True,
                fill_opacity=0.7
            ).add_to(mapa)

    elif tipo_mapa == "HeatMap":

        dados_heatmap = df_brasil[
            ['lat', 'lon', 'potencia_kw']
        ].values.tolist()

        HeatMap(dados_heatmap).add_to(mapa)

    elif tipo_mapa == "Potencial Eólico":

        for _, row in df_brasil.iterrows():

            folium.CircleMarker(
                location=[row['lat'], row['lon']],
                radius=row['potencia_kw'] / 2000,
                popup=f"""
                Estado: {row['estado']}<br>
                Potência: {row['potencia_kw']:.2f} kW
                """,
                color="red",
                fill=True,
                fill_opacity=0.7
            ).add_to(mapa)

    st_folium(
        mapa,
        use_container_width=True,
        height=800
    )

with aba2:
	regiao = st.selectbox(
    	"Estado para previsão",
    	df_brasil["estado"].unique(),
		key="ia_estado"
	)

	df_filtrado = df_brasil[df_brasil["estado"] == regiao]

	previsao_7dias = df_filtrado.iloc[0]["previsao_array"]
	fig = go.Figure()
	fig.add_trace(
    go.Scatter(
        y=previsao_7dias,
        mode='lines',
        name='Previsão IA'
    	)
	)

	fig.update_layout(
    title="Previsão de Vento - Próximos 7 Dias",
    xaxis_title="Horas Futuras",
    yaxis_title="Velocidade do Vento"
	)
	
	st.metric(
    "Previsão IA (7 dias)",
    f"{df_filtrado['vento_7dias'].mean():.2f} m/s"
	)
	st.plotly_chart(
    fig,
    use_container_width=True
)

with aba3:
	st.metric(
		"Maior Potencial",
		 estado_maior,
    	f"{potencia_maior:.2f} kW"
	)
	st.metric(
		"Região mais eficiente",
		melhor_regiao
	)

	st.metric(
		"Média Nacional",
		f"{media_nacional:.2f} kW"
	)

	st.subheader("Ranking Energético")
	st.dataframe(
		ranking[
			["estado", "potencia_kw"]
		]
	)

	st.bar_chart(
		ranking.set_index("estado")["potencia_kw"]
	)





