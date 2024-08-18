#import streamlit
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sas
import plotly.graph_objects as go
import plotly.express as px
import datetime
from datetime import date,timedelta
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller



st.markdown("""
    <style>
        .title {
            background-color: #00FF00; /* Green background */
            color: #000000; /* Black text */
            padding: 10px;
            border-radius: 5px;
            text-align: center;
        }
        .subheader {
            background-color: #FFFF00; /* Yellow background */
            color: #000000; /* Black text */
            padding: 5px;
            border-radius: 5px;
        }
    </style>
    <div class="title">
        <h2>STOCK MARKET FORECASTING APP</h2>
    </div>
    """, unsafe_allow_html=True)
#Title


# add an image from online source
st.image("stock.jpg")

#Add Sub-title
st.subheader("APPLICATION BUILT TO FORECAST THE STOCK MARKET PRICE.")
import streamlit as st

# add an image for sub-title
# st.image("stock2.jpg")


# Take input from the user of app about the start and end date 

#Side bar

st.sidebar.header('SELECT BELOW PERAMETERS')
start_date = st.sidebar.date_input('Start date',date(2023,1,1))
end_date = st.sidebar.date_input('End date',date(2024,8,18))

#Add ticker symbol list
ticker_list = ticker_list = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "BRK.B", "JNJ", "V", 
               "PG", "UNH", "JPM", "XOM", "MA", "LLY", "HD", "ABBV", "MRK", "PEP", 
               "KO", "PFE", "AVGO", "COST", "CSCO", "ADBE", "ACN", "CRM", "NFLX", "ABT", 
               "TMO", "INTC", "DIS", "NEE", "MCD", "NKE", "TXN", "WMT", "LIN", "PM", 
               "AMD", "VZ", "MS", "MDT", "CVX", "UNP", "HON", "ORCL", "CMCSA", "AMGN", 
               "BMY", "SBUX", "DHR", "UPS", "RTX", "CAT", "LOW", "IBM", "AXP", "INTU", 
               "SPGI", "GS", "BLK", "NOW", "LMT", "QCOM", "T", "GE", "SCHW", "ADP", 
               "PLD", "BKNG", "ISRG", "MO", "MMC", "CI", "DE", "DUK", "C", "TGT", 
               "USB", "PYPL", "SYK", "CL", "CB", "ZTS", "BDX", "MSCI", "EL", "APD", 
               "TJX", "ADSK", "EW", "SO", "GILD", "FDX", "FIS", "ALL", "HUM", "MET"]

ticker = st.sidebar.selectbox('Select Company ',ticker_list)

#add image at side 
image = st.sidebar.image("stock2.jpg")


#Fetch data grom User inputs using by Finance Library
data = yf.download(ticker,start = start_date, end = end_date)
#add date as a column to the data frome 
data.insert(0,"Date",data.index,True)
data.reset_index(drop=True,inplace=True)
st.write('Date from',start_date,'to',end_date)
st.write(data)

# Plot the data 

st.header('DATA VISUALIZATION')
st.subheader('Plot of the data')
st.write("Select the date range to display the data based on the selected duration")
st.write('Date from',start_date,'to',end_date)
fig = px.line(data,x="Date",y=data.columns,title='Closing Price of the Stock',width=1000,height=600)
st.plotly_chart(fig)

# add a select box to select a column from data
column = st.selectbox('Select the column to be used for the forecasting',data.columns[1:])

#sub-setting the data
data = data[["Date",column]]
st.write('Selected Data')
st.write(data)

#ADF test check Stationary 
st.header('Is the data Stationary?')
st.write(adfuller(data[column])[1]<0.05)

#lets decompose the data 
st.header('Decomposition of the Data')
decomposition = seasonal_decompose(data[column],model='additive',period=12)
st.write(decomposition.plot())

#Make Same Plot in Plotly
st.write("## Ploting the Decompostion in Plotly")
st.plotly_chart(px.line(x=data["Date"],y=decomposition.trend,title='Trend',width=1000,height=400,labels={'x':"date",'y':"Price",}).update_traces(line_color='blue'))
st.plotly_chart(px.line(x=data["Date"],y=decomposition.seasonal,title='Seasonality',width=1000,height=400,labels={'x':"date",'y':"Price",}).update_traces(line_color='red'))
st.plotly_chart(px.line(x=data["Date"],y=decomposition.resid,title='Residuals',width=1000,height=400,labels={'x':"date",'y':"Price",}).update_traces(line_color='yellow',line_dash = 'dot'))

#Let run the Model
#User input for the three perameters of the model and seasonal order
p=st.slider('Select the value of P',0,5,2)
d=st.slider('Select the value of d',0,5,2)
q=st.slider('Select the value of q',0,5,2)
seasonal_order = st.number_input ('Select the value of Seasonal P', 0,24,12 )

#Training the model

model = sm.tsa.statespace.SARIMAX(data[column],order=(p,d,q),seasonal_order=(p,d,q ,seasonal_order))
model=model.fit()

#Print model Summary 

st.header("Model Summary")
st.write(model.summary())
st.write(".....")

#title Predict the future value forecasting 
st.write("<p style='color:green;font-size:50px;font-weight:bold'>FORECASTING THE DATA</p>", unsafe_allow_html=True)


#Predict the future values (Forecasting)
forecast_period = st.number_input('Select the number of days to forecast',1,365,10)

#Predict the future values 
predictions = model.get_prediction(start=len(data),end=len(data)+forecast_period)
predictions = predictions.predicted_mean
st.write([predictions])

#add index to the predictions

predictions.index= pd.date_range(start=end_date,periods=len(predictions),freq='D')
predictions=pd.DataFrame(predictions)
predictions.insert(0,"Date",predictions.index,True)
predictions.reset_index(drop=True,inplace=True)
st.write("Predicions",predictions)
st.write("Actual Data",data)
st.write("...")

# #lets Plot the Data
fig=go.Figure()
# #add actual data to the Plot
fig.add_trace(go.Scatter(x=data["Date"], y=data[column],mode='lines', name='Actual',line=dict(color='blue')))
# #add predicted data to the Plot
fig.add_trace(go.Scatter(x=predictions["Date"],y=predictions["predicted_mean"],mode='lines',name='predicted',line=dict(color='red')))
# #set the title and axis levels 
fig.update_layout(title='Actual vs Predicted', xaxis_title="Date",yaxis_title="price",width=1200,height=500)
# #display the plot 
st.plotly_chart(fig)

#add buttons to show and hide seprate plots 

show_plots = False
if st.button('Show Seprate Plots'):
    if not show_plots:
        st.write(px.line(x=data["Date"],y=data[column],title='Actual',width=1000,height=400,labels={'x':"Date",'y':"Price",}).update_traces(line_color='blue'))
        st.write(px.line(x=predictions["Date"],y=predictions["predicted_mean"],title="Pridicted",width=1000,height=400,labels={'x':"date",'y':"Price",}).update_traces(line_color='red'))
    show_plots=True
else:
    show_plots=False
#add hide Plot Button
    hide_plots = False
if st.button('Hide Seprate Plots'):
    if not hide_plots:
        hide_plots=True
else:
    hide_plots=False

    st.write("...")


# Display a thank you message
st.markdown("<h2 style='text-align: center;'>Thank You For Using This App</h2>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>Please Share with Others and Connect with my Social Platforms</h4>", unsafe_allow_html=True)

# Define the logos and URLs
github_logo = "https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png"
linkedin_logo = "https://upload.wikimedia.org/wikipedia/commons/0/01/LinkedIn_Logo.svg"
streamlit_logo = "https://streamlit.io/images/brand/streamlit-mark-color.png"

# Display the logos and links in a single row with LinkedIn logo slightly larger
st.markdown(f"""
    <div style="display: flex; align-items: center; justify-content: center; gap: 20px;">
        <a href="https://github.com/meshahan" target="_blank" style="text-decoration: none; color: inherit;">
            <img src="{github_logo}" width="30"/> GitHub
        </a>
        <a href="https://www.linkedin.com/in/ibn-adam-aa5337311/" target="_blank" style="text-decoration: none; color: inherit;">
            <img src="{linkedin_logo}" width="100"/> LinkedIn
        </a>
        <a href="https://share.streamlit.io/user/meshahan" target="_blank" style="text-decoration: none; color: inherit;">
            <img src="{streamlit_logo}" width="30"/> Streamlit
        </a>
    </div>
    """, unsafe_allow_html=True)
