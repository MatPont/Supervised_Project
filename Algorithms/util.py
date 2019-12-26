import sys
import pandas as pd

def balance(seq):
    from collections import Counter
    from numpy import log

    n = len(seq)
    classes = [(clas,float(count)) for clas,count in Counter(seq).items()]
    k = len(classes)

    H = -sum([ (count/n) * log((count/n)) for clas,count in classes]) #shannon entropy
    return H/log(k)
    


if __name__ == "__main__":
    df = pd.read_csv(sys.argv[1], sep="\t", header=None)
    print(df.shape)
    print(balance(df.iloc[:,-1]))
    
    df = pd.read_csv(sys.argv[1], header=None)
    print(df.shape)
    print(balance(df.iloc[:,-1]))    
