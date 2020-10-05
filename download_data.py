from binance.websockets import BinanceSocketManager
from binance.client import Client
import configparser
import json
import sys


def process_message(msg):
    print(f"{msg['T']},{msg['p']},{msg['q']}", flush=True)


config = configparser.ConfigParser()
config.read('secrets/secret-binance-prod.txt')
config = config['Binance']

client = Client(config['api_key'], config['api_secret'])

bm = BinanceSocketManager(client)

print("Starting infinite download, Ctrl-C to stop.")

sys.stdout = open("data/websocket/trades.csv", "w")
print("T,p,q")
conn_key = bm.start_aggtrade_socket('BTCUSDT', process_message)

bm.start()
