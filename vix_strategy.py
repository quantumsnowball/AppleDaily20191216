import scipy.stats, argparse
import pandas as pd, numpy as np, matplotlib.pyplot as plt

class Strategy:
    def __init__(self, **config):
        self.config = config
        for name,value in config.items():
            setattr(self, name, value)  

    def get_close(self, ticker):
        ts = pd.read_csv(f'resources/{ticker}.csv', index_col=0, parse_dates=True)
        close = ts['Adj Close']
        return close

    def signal(self, sr):
        pctl = scipy.stats.percentileofscore(sr, sr[-1])/100
        pctl_inv = 1-pctl
        lv = self.minLv + pctl_inv*(self.maxLv-self.minLv)
        return lv

    def run(self):
        # define date range
        bm = self.get_close(self.benchmark)
        global_start = bm.loc[:self.start].iloc[-self.rollD-self.sma+2:].index[0]
        
        # read indicator
        ind_sr = self.get_close(self.indicator).loc[global_start:self.end]
        sma_ind_sr = ind_sr.rolling(self.sma).mean()
        
        # make signal
        signal = sma_ind_sr.rolling(self.rollD).apply(self.signal, raw=False).dropna()
        target_lv = pd.Series(signal.shift(1).dropna(), name='target_lv')
        
        # calculate portfolio and benchmark nav series
        stk_sr = self.get_close(self.stock).loc[self.start:self.end]
        stk_chg_sr = stk_sr.pct_change()
        porf_chg_sr = stk_chg_sr*target_lv
        porf_nav_sr = pd.Series((1+porf_chg_sr.fillna(0)).cumprod()*self.ininav, name='portfolio')
        porf_bm_sr = pd.Series((1+stk_chg_sr.fillna(0)).cumprod()*self.ininav, name='benchmark')
        account = pd.concat([porf_nav_sr, porf_bm_sr], axis=1)

        # pack
        self.account = account
        self.target_lv = target_lv

        return self

    def evaluate(self, *, show=False):
        def account_metrics(ts):
            chgs = np.log(ts).diff()
            mu = chgs.mean()*252
            sigma = chgs.std()*np.sqrt(252)
            def cal_sharpe(mu, sigma, rf=0.025):
                return (mu - rf)/(sigma)
            sharpe = cal_sharpe(mu, sigma)
            def cal_drawdown(ts):
                run_max = np.maximum.accumulate(ts)
                end = (run_max - ts).idxmax()
                start = (ts.loc[:end]).idxmax()
                low = ts.at[end]
                high = ts.at[start]
                dd = low/high-1
                return dd
            mdd = cal_drawdown(ts)
            def cal_cagr(ts, basis=252):
                cagr = (ts[-1]/ts[0])**(basis/len(ts))-1
                return cagr
            cagr = cal_cagr(ts)
            metrics = {'mu':mu, 'sigma':sigma, 'sharpe':sharpe, 'mdd':mdd, 'cagr':cagr, }
            return metrics
        metrics = {name:account_metrics(ts) for name,ts in self.account.items()}        
        def plot():
            fig, ax = plt.subplots(2,1, figsize=(15,9), sharex=True, constrained_layout=True,
                            gridspec_kw={'width_ratios':[1], 'height_ratios':[3,1]})            
            start, end = self.account.index[0].date(), self.account.index[-1].date()
            return_st = self.account['portfolio'][-1]/self.account['portfolio'][0]-1
            return_bm = self.account['benchmark'][-1]/self.account['benchmark'][0]-1
            metrics_st = ", ".join([f"{k}:{v:.2%}" for k,v in metrics["portfolio"].items()])
            metrics_bm = ", ".join([f"{k}:{v:.2%}" for k,v in metrics["benchmark"].items()])
            maintitle = (f'Performance, start: {start}, end:{end}\n'
                        f'Strategy : {return_st:.2%} ({metrics_st})\n'
                        f'Benchmark: {return_bm:.2%} ({metrics_bm})\n')
            ax[0].set_title(maintitle)
            ax[0].plot(self.account['portfolio'], label='startegy')
            ax[0].plot(self.account['benchmark'], label=f'benchmark({self.benchmark})')
            ax[0].legend()
            subtitle = (f'Target leverage\nSignal: {self.indicator} percentile score, '
                        f'rollD:{self.rollD}, SMA:{self.sma}, minLv:{self.minLv}, maxLv:{self.maxLv}')
            ax[1].set_title(subtitle)
            ax[1].plot(self.target_lv)
            plt.show(block=True)            
        if show: plot()
        
        return {
            'config':self.config,
            'metrics':metrics
        }

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-s', '--start', type=str, default='20000101', help='Start date of backtest')
    parser.add_argument('-e', '--end', type=str, default=None, help='End date of backtest')
    parser.add_argument('-stk', '--stock', type=str, default='SPY', help='Stock to trade')
    parser.add_argument('-bm', '--benchmark', type=str, default='SPY', help='Stock as benchmark')
    parser.add_argument('-ind', '--indicator', type=str, default='^VIX', help='Indicator to decide leverage')
    parser.add_argument('-rD', '--rollD', type=int, default=50, help='Rolling window in days to look back')
    parser.add_argument('--sma', type=int, default=100, help='Smooth the indicator by n-days simple moving average')
    parser.add_argument('-minL', '--minLv', type=float, default=0.5, help='Min leverage level')
    parser.add_argument('-maxL', '--maxLv', type=float, default=1.8, help='Max leverage level')
    parser.add_argument('--ininav', type=int, default=1e7, help='Initial nav')
    sargs = parser.parse_args()
    
    backtest = Strategy(**vars(sargs))
    backtest.run()
    backtest.evaluate(show=True)

if __name__ == '__main__':
    main()