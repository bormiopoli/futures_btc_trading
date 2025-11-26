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
- **Multi-timeframe Technical Analysis** (3, 14, 28 periods)
- **Ensemble ML Model** with pre-trained neural network
- **Automated Position Management** with risk controls
- **Real-time Market Data** from Binance API

### 📊 Technical Indicators
- **Trend**: MACD, ADX, Aroon, Ichimoku
- **Momentum**: RSI, Stochastic Oscillator  
- **Volatility**: ATR, Bollinger Bands
- **Volume**: Accumulation/Distribution Index

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

### Prerequisites

```bash

# Extract
pip install -r requirements.txt
```