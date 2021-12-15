import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import find_peaks


COL_TIME = 0
COL_PRICE = 1
COL_QUANTITY = 2
COL_CNT = 3


def stds(prices, window_width):
    stds = []
    mns = []
    for i in range(window_width, prices.shape[0]-window_width):
        win = prices[i-window_width:i]
        stds.append(np.std(win))
        mns.append(np.mean(win))

    for i in range(window_width):
        stds.insert(0,stds[0])

    return mns, stds

def speeds (numpy_data, window_width):
    spds=[]

    for i in range(window_width+1, prices.shape[0]-window_width):
        dp = numpy_data[i-window_width:i, COL_PRICE]  - numpy_data[i-1-window_width:i-1, COL_PRICE]
        dt = numpy_data[i-window_width:i, COL_TIME]  - numpy_data[i-1-window_width:i-1, COL_TIME]
        v = dp/dt
        v = v[~np.isnan(v) & ~np.isinf(v)]
        spds.append(np.mean(v))

    for i in range(window_width+1):
        spds.insert(0,spds[0])

    return spds


if __name__ == '__main__':

    # train_df = pd.read_csv('data/websocket/trades.csv')
    print("LOL")
    train_df = pd.read_csv('data/websocket/historical-reversed-trades.csv', nrows = 5000000)
    print("Read.")
    train_df.sort_values(by=['id'], inplace=True)
    train_df.drop('id', axis=1, inplace=True)
    print(train_df)
    ps = 900000
    pivot =ps+200000
    cutoff = -1#2*pivot
    sample_window_width = 50
    check_width = 1500

    train_data = train_df[ps:pivot].to_numpy()
    # test_data = train_df[pivot:cutoff].to_numpy()

    prices = train_data[:, COL_PRICE]
    vols = train_data[:, COL_QUANTITY]



    ww=10000
    delay=10

    exittimeout=2000

    tot = 0
    winlong=0

    treaderesults = []

    for i in range(ww):
        treaderesults.append(0)
    
    for i in range(ww, prices.shape[0]-ww):
        winopenprice = prices[i-ww]
        wincloseprice = prices[i]
        nextpwin = prices[i+delay]
        #print(winopenprice, wincloseprice, (wincloseprice-winopenprice), nextpwin)

        tot+=1
        nextw = prices[i+delay:i+delay+exittimeout]

        tgtlong = (wincloseprice - winopenprice)*2 + wincloseprice
        if(np.any( nextw> wincloseprice) and np.all(nextw > winopenprice) and np.any(nextw>tgtlong)) :
            winlong+=1
            treaderesults.append(1)
        elif (np.any( nextw> wincloseprice) and np.any(nextw <= winopenprice)):
            treaderesults.append(-1)
        else:
            treaderesults.append(0)



    print(tot, winlong)
    treaderesults=np.array(treaderesults)


    print("facepalm")
    m,s = stds(prices, ww)
    print("ilekurewa")
    # mozna = speeds(train_data, ww)
    # print("mozna")
    # liczyc = stds(prices, 10000)
    # print("liczyc")

    s = np.array(s)
    # ds = np.maximum((s[1:] / s[:-1]) / 1.01, 1) 

    xs=[]
    ys=[]

    for i in range(ww,treaderesults.shape[0]-ww):
        xs.append(s[i])
        ys.append(treaderesults[i])
        



    fig, axs = plt.subplots(4)
    fig.suptitle('Vertically stacked subplots')
    axs[0].plot(prices)
    axs[1].plot(s)
    axs[2].plot(treaderesults)
    axs[3].plot(xs,ys)
    # axs[2].plot(mozna*s)
    #axs[2].plot(np.mean(s))
    #axs[2].hlines(y=np.mean(s), xmin=-100, xmax=10000)




    # peaks2, _ = find_peaks(ile, prominence=1) 
    # axs[2].plot(peaks2)


    # #x=ile
    # x = np.sin(2*np.pi*(2**np.linspace(2,10,1000))*np.arange(1000)/48000) + np.random.normal(0, 1, 1000) * 0.15
    # x=np.array(ile)
    # print(x, x.shape)
    # peaks, _ = find_peaks(x, distance=20)
    # peaks2, _ = find_peaks(x, prominence=1)      # BEST!
    # peaks3, _ = find_peaks(x, width=20)
    # peaks4, _ = find_peaks(x, threshold=0.4)     # Required vertical distance to its direct neighbouring samples, pretty useless
    # plt.subplot(2, 2, 1)
    # plt.plot(peaks, x[peaks], "xr"); plt.plot(x); plt.legend(['distance'])
    # plt.subplot(2, 2, 2)
    # plt.plot(peaks2, x[peaks2], "ob"); plt.plot(x); plt.legend(['prominence'])
    # plt.subplot(2, 2, 3)
    # plt.plot(peaks3, x[peaks3], "vg"); plt.plot(x); plt.legend(['width'])
    # plt.subplot(2, 2, 4)
    # plt.plot(peaks4, x[peaks4], "xk"); plt.plot(x); plt.legend(['threshold'])
    # plt.show()




    # axs[2].plot(mozna)
    # axs[3].plot(liczyc)

    # plt.plot(prices)
    # plt.plot(vols)
    plt.show()