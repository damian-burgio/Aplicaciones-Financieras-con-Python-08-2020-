"""
APLICACIONES FINANCIERAS EN PYTHON
FECHA: 20/08/2020
PROFESOR: MARTIN CONOCCHIARI
"""

#############################################################################
###############  DATOS FINANCIEROS Y SU PROCESAMIENTO  ######################
#############################################################################

#Importo librerias (e instalo alguna si hace falta)
#pip install yfinance
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import seaborn as sns 
import scipy.stats as scs
import statsmodels.api as sm
#import statsmodels.tsa.api as smt

#Descargamos la data
df_yahoo = yf.download('AAPL')
df_yahoo1 = yf.download('AAPL',start='2010-01-01',end='2020-08-19',progress=True)
df_yahoo2 = yf.download(['AAPL'],start='2010-01-01',end='2020-08-19',
                       auto_adjust=True,actions='inline',progress=True)
df_yahoo3 = yf.download(['AAPL','MSFT','AMZN'],interval="1h",
                       auto_adjust=True,actions='inline',progress=True,period="1mo")
df_yahoo4 = yf.download(['AAPL','MSFT','AMZN'],start='2010-01-01',end='2020-08-19',
                       auto_adjust=True,actions='inline',progress=True)

#Si auto_adjust=True descargo precios ajustados.
#Si queremos descargar pago dedividendos and stock splits agregar actions='inline'.
#se podria hacer dsd quandl o intrinio, pero hay que generar una clave para la API.

#si quisiera exportar la data a un excel
df_yahoo2.to_excel(r'C:\Users\marti\Desktop\CursoPython\df_yahoo2.xlsx', index = True)

############################################################################
'''
Si quisiera obtener por ejemplo los valores de todos los activos el S&P500
'''
import bs4 as bs
import requests
import yfinance as yf
import datetime

resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
soup = bs.BeautifulSoup(resp.text, 'lxml')
table = soup.find('table', {'class': 'wikitable sortable'})
tickers = []
for row in table.findAll('tr')[1:]:
    ticker = row.findAll('td')[0].text
    tickers.append(ticker)

tickers = [s.replace('\n', '') for s in tickers]
start = datetime.datetime(2019,1,1)
end = datetime.datetime(2019,7,17)
todosyp500 = yf.download(tickers, start=start, end=end)
print(todosyp500)
##############################################################################

#Me quedo solo con la Adj Close y le cambio el nombre a adj_close
df = df_yahoo1
df = df.loc[:, ['Adj Close']]
df.rename(columns={'Adj Close':'adj_close'}, inplace=True)

#calculo los retornos simples y logaritmicos de los adj_close
df['simple_rtn'] = df.adj_close.pct_change() #pct_change es un metodo de pandas
df['log_rtn'] = np.log(df.adj_close/df.adj_close.shift(1))
df.head() #para ver los primeros resultados del output. Puedo definir la cant con n=
df.tail() #para ver los últimos resultados del output

#me quedo con la columna de los retornos logaritmicos
df = df.loc[:, ['log_rtn']]
df.dropna(axis=0, inplace=True) #elimino los na
df.head()

#grafico los retornos log de AAPL
plt.plot(df)
plt.ylabel('variacion')
plt.xlabel('periodo')
plt.show()

#Defino la funcion para calcular la volatilidad realizada
def realized_volatility(x):
    return np.sqrt(np.sum(x**2)) #raiz de los retornos al cuadrado

#Calculo la volatilidad realizada mensual
df_rv = df.groupby(pd.Grouper(freq='M')).apply(realized_volatility)
df_rv.rename(columns={'log_rtn': 'rv'}, inplace=True)

#Anualizo los valores
df_rv.rv = df_rv.rv * np.sqrt(12) #es 12 xq los valores son mensuales

#Grafico los resultados
fig, ax = plt.subplots(2, 1, sharex=True)
ax[0].plot(df)
ax[1].plot(df_rv)
plt.show()

#Visualizing time series data
# download data as pandas DataFrame
df = yf.download('MSFT', auto_adjust = False, progress=False)
df = df.loc[:, ['Adj Close']]
df.rename(columns={'Adj Close': 'adj_close'}, inplace=True)


'''
Identificación de outliers
utilizando el metodo de 3σU
'''
#Traigo los datos y calculos retornos log
df = df_yahoo1
df = df.loc[:, ['Adj Close']]
df.rename(columns={'Adj Close':'adj_close'}, inplace=True)
df['log_rtn'] = np.log(df.adj_close/df.adj_close.shift(1))
df.head()

#Calculo la media y desvio estandar, móviles
df_rolling = df[['log_rtn']].rolling(window=21) \
                               .agg(['mean', 'std'])
df_rolling.columns = df_rolling.columns.droplevel()

#Uno las metricas móviles a las originales (tabla df)
df_outliers = df.join(df_rolling)

#Defino la funcion para detectar outliers
def indentify_outliers(row, n_sigmas=3):
    '''  
    Parametros
    ----------
    row : row con los retornos
    n_sigmas : cant de desvios para detectar outliers
        
        
    Returns de la funcion
    -------
    0/1 : un entero que vale 1 indicando un outlier, 0 en cualquier otro caso
    '''
    x = row['log_rtn']
    mu = row['mean']
    sigma = row['std']
    
    if (x > mu + 3 * sigma) | (x < mu - 3 * sigma):
        return 1
    else:
        return 0

#Identificado los outliers y los extrae para usarlos luego
df_outliers['outlier'] = df_outliers.apply(indentify_outliers,
                                           axis=1)
outliers = df_outliers.loc[df_outliers['outlier'] == 1, 
                           ['log_rtn']]
#Grafico los resultados
fig, ax = plt.subplots()

ax.plot(df_outliers.index, df_outliers.log_rtn, 
        color='blue', label='Normal')
ax.scatter(outliers.index, outliers.log_rtn, 
           color='red', label='Outlier')
ax.set_title("Apple's stock returns")
ax.legend(loc='lower right')

plt.show()


'''
Analisis de normalidad de una serie financiera
'''

#Descargo la data del indice S&P 500 y calculo sus retornos
df = yf.download('^GSPC')

df = df[['Adj Close']].rename(columns={'Adj Close': 'adj_close'})
df['log_rtn'] = np.log(df.adj_close/df.adj_close.shift(1))
df = df[['adj_close', 'log_rtn']].dropna(how = 'any')

#Calculo la función de densidad de probabilidad Normal utilizando
#la media y el desvio de los retornos observados
r_range = np.linspace(min(df.log_rtn), max(df.log_rtn), num=1000)
mu = df.log_rtn.mean()
sigma = df.log_rtn.std()
asimetria = scs.skew(df.log_rtn)
curtosis = scs.kurtosis(df.log_rtn)
norm_pdf = scs.norm.pdf(r_range, loc=mu, scale=sigma)

#Grafico histograma y Q-Q Plot
fig, ax = plt.subplots(1, 2, figsize=(16, 8))

#Histograma
sns.distplot(df.log_rtn, kde=False, norm_hist=True, ax=ax[0])                                    
ax[0].set_title('Distribucion retornos S&P500', fontsize=16)                                                    
ax[0].plot(r_range, norm_pdf, 'g', lw=2, 
           label=f'N({mu:.2f}, {sigma**2:.4f})')
ax[0].legend(loc='upper left');

# Q-Q plot
qq = sm.qqplot(df.log_rtn.values, line='s', ax=ax[1])
ax[1].set_title('Q-Q plot', fontsize = 16)

plt.tight_layout()
plt.show()

#Resumen medidas estadisticas de los retornos log
jb_test = scs.jarque_bera(df.log_rtn.values)

print('---------- Estadística Descriptiva ----------')
print('Range of dates:', min(df.index.date), '-', max(df.index.date))
print('Number of observations:', df.shape[0])
print(f'Media: {df.log_rtn.mean():.4f}')
print(f'Mediana: {df.log_rtn.median():.4f}')
print(f'Min: {df.log_rtn.min():.4f}')
print(f'Max: {df.log_rtn.max():.4f}')
print(f'Desvio Estandar: {df.log_rtn.std():.4f}')
print(f'Asimetria: {df.log_rtn.skew():.4f}')
print(f'Kurtosis: {df.log_rtn.kurtosis():.4f}') 
print(f'Estadistico Jarque-Bera: {jb_test[0]:.2f} with p-value: {jb_test[1]:.2f}')

#############################################################################
#########################  PORTAFOLIOS OPTIMOS  #############################
#############################################################################

#Importo librerias (e instalo alguna si hace falta)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.optimize import minimize
import statsmodels.api as sm
import scipy.stats as scs

#Seteo los parametros
N_PORTFOLIOS = 10000 #cantidad de portafolios simulados
N_DAYS = 252
tickers = ['AAPL','AMZN','GE','JNJ']
START_DATE = '2010-07-01'
END_DATE = '2020-08-19'
n_assets = len(tickers)

#Descargamos la data
data = yf.download(tickers, start = START_DATE, end = END_DATE, progress=True)
data = data.dropna()
data = data.loc[:, ['Close']]
data = data.xs('Close', axis=1) #uso esto para borrar el doble titulo de cada columna
data.head()

data.to_excel(r'C:\Users\marti\Google Drive\Curso Python\data.xlsx', 
                   index = True)

#graficos la evolucion de los precios de los activos a considerar
data.plot(title='Precios de los activos considerados')

#muestro las n series temporales, pero normalizados para que empiecen en 100
(data / data.iloc[0] * 100).plot(figsize=(8, 6), grid=True,title='Precios de los activos considerados')

#calculo retornos logaritmicos
ret_l = np.log(data/data.shift(1))
ret_l = ret_l.dropna()
ret_l.head()

ret_l.plot(title='Retorno diario de los activos considerados')
ret_l.hist(bins=50, figsize=(9, 6))

#info estadistica adicional de cada activo
#defino funcion que luego voy a usar
def asim_y_curt(arr):  
    print("Skew of data set  %14.3f" % scs.skew(arr))
    print("Kurt of data set  %14.3f" % scs.kurtosis(arr))
      
for tck in tickers:
    print("\nResults for ticker %s" % tck)
    print(32 * "-")
    log_data = np.array(ret_l[tck].dropna())
    asim_y_curt(log_data)

#Retorno de cada activo
ret_prom = ret_l.mean()
ret_act1 = ret_prom[0] * N_DAYS
ret_act2 = ret_prom[1] * N_DAYS
ret_act3 = ret_prom[2] * N_DAYS
ret_act4 = ret_prom[3] * N_DAYS

#Matriz de varianzas y covarianzas
mat_cov = ret_l.cov()
desvios = ret_l.std()
desvio_act1 = desvios[0] * np.sqrt(N_DAYS)
desvio_act2 = desvios[1] * np.sqrt(N_DAYS)
desvio_act3 = desvios[2] * np.sqrt(N_DAYS)
desvio_act4 = desvios[3] * np.sqrt(N_DAYS)

#Primer intento al azar para construir un portfolio y encontrar su Sharpe Ratio
np.random.seed()
# Columna de Acciones
print('Acciones')
print(data.columns)
print('\n')
# Genero numeros random usando la distribución uniforme (0 a 1)
print('Numeros Aleatorios')
weights = np.array(np.random.random(n_assets))
print(weights)
print('\n')
# Transformo los numeros aleatorios a % (Cada numero / La suma de los n numeros)
print('Rebalanceo para que de 1')
weights = weights / np.sum(weights)
print(weights)
print('\n')
# Retorno esperado anual del portafolio
print('Retorno esperado del Portfolio')
exp_ret = np.sum(ret_prom * weights) * N_DAYS
print(exp_ret)
print('\n')
# Volatilidad esperada anual
print('Volatilidad Esperada')
exp_vol = np.sqrt(np.dot(weights.T, np.dot(mat_cov * N_DAYS, weights)))
print(exp_vol)
print('\n')
# Sharpe Ratio
SR = exp_ret/exp_vol
print('Sharpe Ratio')
print(SR)

#Comenzamos las simulaciones de portafolios inicializando las matrices
all_weights = np.zeros((N_PORTFOLIOS,len(data.columns)))
ret_arr = np.zeros(N_PORTFOLIOS)
vol_arr = np.zeros(N_PORTFOLIOS)
sharpe_arr = np.zeros(N_PORTFOLIOS)
#seq = list(range(0,N_PORTFOLIOS)) #lista que arranca en 0 y termina en cant de port 

#Por cada portafolio de las N simulaciones, calculamos las ponderaciones, 
#el retorno, la volatilidad y el SR.
for i in range(N_PORTFOLIOS):
    weights=np.array(np.random.random(n_assets))
    weights=weights/np.sum(weights)
    all_weights[i,:] = weights
    ret_arr[i] = np.sum((ret_prom * weights) * N_DAYS)
    vol_arr[i] = np.sqrt(np.dot(weights.T, np.dot(mat_cov * N_DAYS, weights)))
    
for i in range(N_PORTFOLIOS):
    sharpe_arr[i] =  ret_arr[i]/vol_arr[i]
    
#Analicemos los resultados
print("El Sharpe Ratio Max es: " + str(sharpe_arr.max()))
print("en el Portfolio número: " + str(sharpe_arr.argmax()))
print("La Menor Volatilidad es: " + str(vol_arr.min()))
print("en el Portfolio número: " + str(vol_arr.argmax()))
print()
print('Proporciones:\n')
print("APPLE: " + str(round(all_weights[sharpe_arr.argmax(),:][0]*100,2)) + "%")
print("AMAZON: " + str(round(all_weights[sharpe_arr.argmax(),:][1]*100,2)) + "%")
print("GENERAL ELECTRIC: " + str(round(all_weights[sharpe_arr.argmax(),:][2]*100,2)) + "%")
print("JOHNSON & JOHNSON: " + str(round(all_weights[sharpe_arr.argmax(),:][3]*100,2)) + "%\n")

#Graficamente
max_sr_ret = ret_arr[sharpe_arr.argmax()]
max_sr_vol = vol_arr[sharpe_arr.argmax()]
min_vol = vol_arr[vol_arr.argmin()]
ret_min_vol = ret_arr[vol_arr.argmin()]

plt.figure(figsize=(20,10))
plt.scatter(vol_arr,ret_arr,c=sharpe_arr,cmap='RdYlGn')
plt.colorbar(label='Sharpe Ratio')
plt.xlabel('Volatilidad', fontsize=25, color='black')
plt.ylabel('Retorno', fontsize=25, color='black')
plt.title('Portafolios simulados: ',fontsize=25, color='black')
plt.scatter(max_sr_vol,max_sr_ret,c='red',marker='*',s=250,label='Max Sharpe Ratio')
plt.scatter(min_vol,ret_min_vol,c='black',marker='X',s=250,label='Min Vol')
plt.scatter(desvio_act1,ret_act1,c='black',marker='v',s=200,label='AAPL')
plt.scatter(desvio_act2,ret_act2,c='black',marker='>',s=200,label='AMZN')
plt.scatter(desvio_act3,ret_act3,c='black',marker='d',s=200,label='GE')
plt.scatter(desvio_act4,ret_act4,c='black',marker='h',s=200,label='JNJ')
plt.tick_params(axis='both', colors='blue', labelsize=20)
plt.legend(loc='center right', fontsize=17)
plt.show()

#Resolución matemática
#Función que reciba las proporciones del portafolio y devuelva el retorno, 
#la volatilidad y el SR
def get_ret_vol_sr(weights):
    weights = np.array(weights)
    ret = np.sum(ret_l.mean() * weights) * N_DAYS
    vol = np.sqrt(np.dot(weights.T, np.dot(ret_l.cov() * N_DAYS, weights)))
    sr = ret/vol
    return np.array([ret,vol,sr])

#Función a minimizar, Sharpe Ratio Negativo
def neg_sharpe(weights):
    return  get_ret_vol_sr(weights)[2] * -1

#Restricción para minimizar
def check_sum(weights):
    return np.sum(weights) - 1

cons = ({'type':'eq','fun': check_sum})

#Las proporciones deben estar entre 0 y 1, no permito ir short.
bounds = ((0, 1), (0, 1), (0, 1), (0, 1))

#Optimizamos el portafolio
init_guess = [0.25,0.25,0.25,0.25]
opt_results = minimize(neg_sharpe,init_guess,method='SLSQP',bounds=bounds,constraints=cons)

#Resultados
print('Portafolio Optimo (Periodo Mayo 2010 - Julio 2020):\n')
print('Proporciones:\n')
print("APPLE: " + str(round(opt_results.x[0]*100,2)) + "%")
print("AMAZON: " + str(round(opt_results.x[1]*100,2)) + "%")
print("GENERAL ELECTRIC: " + str(round(opt_results.x[2]*100,2)) + "%")
print("JOHNSON & JOHNSON: " + str(round(opt_results.x[3]*100,2)) + "%\n")
print('Metricas:\n')
print("RETORNO MEDIO: " + str(round(get_ret_vol_sr(opt_results.x)[0]*100,2)) + "%")
print("VOLATILIDAD: " + str(round(get_ret_vol_sr(opt_results.x)[1]*100,2)) + "%")
print("SHARPE: " + str(get_ret_vol_sr(opt_results.x)[2]))

#############################################################################
################## UTILIZO LIBRERIA PYPORTFOLIOOPT  #########################
#############################################################################

#pip install pyfolio
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import plotting

df = data

# Calculo los media y matriz de var-covar anualizada
mu = expected_returns.mean_historical_return(df)
Sigma = risk_models.sample_cov(df)

'''
Portafolio con Max Sharpe
'''
# Calcula la frontera eficiente con mu y sigma
ef = EfficientFrontier(mu, Sigma)
raw_weights = ef.max_sharpe() #optimiza para el Max Sharpe

# 
cleaned_weights = ef.clean_weights() #redondea los weights y elimina los cercanos a cero

# Devuelve los valores de performance
ef.portfolio_performance(verbose=True) #calcula el retorno esperado, volatilidad, 
                                        #y el Sharpe para el Port Optimo

pypfopt.plotting.plot_covariance(Sigma, plot_correlation=False, show_tickers=True)

cla = pypfopt.cla.CLA(mu, Sigma, weight_bounds=(0, 1))
pypfopt.plotting.plot_efficient_frontier(cla, points=100, show_assets=True)

pypfopt.plotting.plot_weights(raw_weights)

'''
Portafolio de Minima varianza de la frontera eficiente
'''
# Calcula la frontera eficiente con mu y sigma
ef = EfficientFrontier(mu, Sigma)
raw_weights = ef.min_volatility() #optimiza para el Min Volatilidad


cleaned_weights = ef.clean_weights()

ef.portfolio_performance(verbose=True)

'''
Optimizando segun un objetivo de retorno o riesgo
'''
# Calcula la frontera eficiente con mu y sigma
ef = EfficientFrontier(mu, Sigma)

# Selecciona el retorno optimo para un riesgo objetivo
ef.efficient_risk(0.2) 

# Selecciona el riesgo minimo para un retorno objetivo
ef.efficient_return(0.2) #minimises risk for a given target return

'''
Retorno y covarianza con ponderacion exponencial
'''
# Promedio ponderado exponencialmente
mu_ema = expected_returns.ema_historical_return(df,
span=252, frequency=252)
print(mu_ema)

# Covarianza ponderada exponencialmente
Sigma_ew = risk_models.exp_cov(df, span=180, frequency=252)

'''
Maximum Sharpe portfolio
'''
# Calcula la frontera eficiente con mu y sigma
ef_ema = EfficientFrontier(mu_ema, Sigma_ew)
raw_weights = ef_ema.max_sharpe() #optimiza para el Max Sharpe

#
cleaned_weights_ema = ef_ema.clean_weights() 

# Devuelve los valores de performance
ef_ema.portfolio_performance(verbose=True)

'''
Portafolio de Minima varianza de la frontera eficiente
'''

# Calcula la frontera eficiente con mu y sigma
ef_ema = EfficientFrontier(mu_ema, Sigma_ew)
raw_weights_ema = ef_ema.min_volatility()

cleaned_weights_ema = ef_ema.clean_weights()

ef_ema.portfolio_performance(verbose=True)

'''
Semicovarianza
'''
Sigma_semi = risk_models.semicovariance(df,
benchmark=0, frequency=252)
print(Sigma_semi)

