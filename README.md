# 🚀 YoBOT Futures Trading System

A sophisticated algorithmic trading system for cryptocurrency futures that uses machine learning and technical analysis to execute automated trades on dYdX.

![Trading Bot](https://img.shields.io/badge/Algorithmic-Trading-blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-green)
![Status](https://img.shields.io/badge/Status-Active-success)

## 📋 Overview

This system combines real-time market data from Binance with advanced technical analysis and machine learning to execute automated futures trades on dYdX with comprehensive monitoring and notification capabilities.

## 🏗️ Architecture

yobot-futures/
├── main.py # Core trading loop & position management
├── functions.py # Technical analysis & ML prediction engine
├── dydx_v4_connection.py # dYdX exchange integration
├── notifications.py # Gmail notification service
├── logger.py # Logging configuration
├── binance_connection.py # Binance market data API
└── plots/ # Generated performance charts

## ✨ Key Features

### 🤖 Trading Strategy
- **Multi-timeframe Technical Analysis**
- **Automated Position Management** with risk controls
- **Real-time Market Data** from Binance API

### 📊 Technical Indicators
## 📊 Technical Analysis Engine

The system employs a sophisticated multi-timeframe technical analysis approach with **3 distinct period configurations** for comprehensive market analysis:

### 🎯 Core Indicator Categories

#### 1. Trend Indicators

- **MACD (3, 14, 28 periods)** - Moving Average Convergence Divergence
  - **Fast periods**: 3, 14, 28
  - **Slow periods**: 14, 28, 56  
  - **Signal periods**: 3, 3, 14

- **ADX (3, 14, 28 periods)** - Average Directional Index
  - Measures trend strength across multiple timeframes

- **Aroon Indicator (3, 14, 28 periods)**
  - Identifies trend changes and strength

- **Ichimoku Cloud (3, 14, 28 periods)**
  - Comprehensive trend, support/resistance, and momentum
  - **Configurations**: (1,3,14), (3,14,28), (14,28,56)

#### 2. Momentum Oscillators

- **RSI (3, 14, 28 periods)** - Relative Strength Index
  - Overbought/oversold conditions across short, medium, and long terms

- **Stochastic Oscillator (3, 14, 28 periods)**
  - **Smooth windows**: 1, 3, 3 periods respectively
  - Momentum and reversal signals

#### 3. Volatility Indicators

- **Average True Range - ATR (3, 14, 28 periods)**
  - Volatility measurement and position sizing

#### 4. Volume Analysis

- **Accumulation/Distribution Index**
  - Money flow based on price and volume

### 🧠 Machine Learning
- **Multi-head Neural Network** for time series forecasting
- **Feature Engineering** with 30+ technical indicators
- **Real-time Prediction** updates every minute
- **Performance Backtesting** with comprehensive analytics

## 🚀 Quick Start

### Setup:
Create a file named binance_credentials.py and write in it the following:
```
BINANCE_API_KEY = "your_binance_api_key"
BINANCE_SECRET = "your_binance_secret"
MY_SECRET = ""
```
Place the file at the root level of the repository.

Then, create a client.json file from Google OAUTH 2.0 page (eg. https://console.cloud.google.com/auth/clients)
rename the file client_secret.json and place it under the folder named file_folder
The client_secret during the first run will be used to create an authentication token, therefore check the terminal where the scrip runs as a link to confirm the authentication for google gmail (notifications) will appear.
### Prerequisites

```bash
 
pip install -r requirements.txt
```

### Run
To run the software execute the file main.py on a python interpreter with dependencies installed, eg.
```
python main.py
```