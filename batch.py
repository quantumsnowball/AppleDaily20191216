import itertools
import pandas as pd
from vix_strategy import Strategy

def main():
    var_fields = {
        'minLv':[.5,.8,1.1,1.3],
        'maxLv':[1.5,1.8,2.1,2.5],
        'rollD':[10,20,50,100,250,500],
        'sma':[5,20,50,100],
    }    
    combos = [{
        'start': '20100101',
        'end': None,
        'stock': 'SPY',
        'benchmark': 'SPY',
        'indicator': '^VIX',        
        'minLv': minLv,
        'maxLv': maxLv,
        'rollD': rollD,
        'sma': sma,
        'ininav': 1.0e7,
    } for minLv,maxLv,rollD,sma in itertools.product(*tuple(var_fields.values()))]

    def trial(combo, i, n):        
        result = Strategy(**combo).run().evaluate()
        cur = {n:result['config'][n] for n in var_fields.keys()}
        print(f'{i+1: >5} / {n} | config: {cur}', end=' | ')
        return result

    results = []
    for i,combo in enumerate(combos):
        try:
            results.append(trial(combo, i, len(combos)))
            print('Success', end='       \r')
        except Exception as e:
            print(str(e))
            continue

    def present(results):
        all = []
        for res in results:
            data = {}
            for key,val in res['config'].items():
                data[key] = val
            for key,val in res['metrics']['portfolio'].items():
                data[key] = val
            for key,val in res['metrics']['benchmark'].items():
                data[key+'_bm'] = val
            all.append(data)
        df = pd.DataFrame(all).sort_values(by='sharpe', ascending=False).reset_index(drop=True)
        print(df.head(10))
        df.to_excel('results.xlsx')
        return df
    df = present(results)
    br()

if __name__ == '__main__':
    main()