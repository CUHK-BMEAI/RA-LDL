import pandas as pd
import argparse
from trainer import train

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str)
    a=parser.parse_args()
    exps=pd.read_csv('./args/'+a.d+'.csv')
    args = exps.iloc[0].to_dict()
    args['seed']=[args['seed']]
    args['device']=[args['device']]
    args['do_not_save']=False
    train(args)

if __name__ == '__main__':
    main()