import backtrader as bt
import pandas as pd
import datetime as dt
import FinanceDataReader as fdr
import backtrader.feeds as btfeeds

class DailyBuy(bt.Strategy):
    def __init__(self):
        pass
    def next(self):
        self.buy(size=10)
df = fdr.DataReader('005930','2024-01-02','2025-02-28')
df.to_csv('005930.csv')

class customCSV(btfeeds.GenericCSVData):
    params=(
        ('dtformat', '%Y-%m-%d'), 
        ('datetime', 0), ('time', -1),
        ('open', 1), ('high', 2), 
        ('low', 3), ('close', 4),
        ('volume', 5), ('openinterest', -1), 
    )

data = customCSV(dataname = '005930.csv')

cerebro = bt.Cerebro()
cerebro.adddata(data)
cash = cerebro.broker.setcash(300000000)
cerebro.addstrategy(DailyBuy)

back_init = cerebro.broker.getvalue()

back_init = cerebro.broker.getvalue()

cerebro.run()

result = cerebro.broker.getvalue()

percentage_increase = ((result - back_init) / back_init) * 100
print(percentage_increase)

#cerebro.plot()