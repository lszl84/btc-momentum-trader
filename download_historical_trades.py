from binance.websockets import BinanceSocketManager
from binance.client import Client
import configparser
import json
import sys
import time

def process_message(msg):
    print(f"{msg['id']},{msg['time']},{msg['price']},{msg['qty']}", flush=True)


config = configparser.ConfigParser()
config.read('secrets/secret-binance-prod.txt')
config = config['Binance']

client = Client(config['api_key'], config['api_secret'])

print("Starting infinite download, Ctrl-C to stop.")

id = None
k = 400

sys.stdout = open("data/websocket/historical-reversed-trades.csv", "w")
print("id,T,p,q")

while True:
    resp = client.get_historical_trades(symbol="BTCUSDT", limit=k, fromId=id)
    for m in reversed(resp):
        process_message(m)
    time.sleep(.3)

    id = int(resp[0]['id'])-k

# bm = BinanceSocketManager(client)

# conn_key = bm.start_aggtrade_socket('BTCUSDT', process_message)

# bm.start()
