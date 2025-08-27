import pandas as pd
import numpy as np

def ma_crossover(df, price = "Close", window_short = 5, window_long = 20):
    d = df.copy()
    ms_name, ml_name = f"MA_{window_short}", f"MA_{window_long}"

    ms, ml = d[ms_name], d[ml_name]
    pms, pml = ms.shift(1), ml.shift(1)

    valid = ms.notna() & ml.notna() & pms.notna() & pml.notna()

    gold = valid & (pms <= pml) & (ms > ml)
    dead = valid & (pms >= pml) & (ms < ml)

    d.loc[gold, "MA_signal"] += 1
    d.loc[dead, "MA_signal"] -= 1

    return d

def ma_strategy(df, ma_signals, price = "Close", max_lots = None):
    """ma windows -> [(ma_short, ma_long)]"""
    d = df.copy()
    for signal in ma_signals:
        short_window, long_window = signal
        d = ma_crossover(d, window_short=short_window, window_long=long_window)

    ev = pd.to_numeric(d["MA_signal"], errors="coerce").fillna(0).astype(int).values

    if max_lots is None:
        max_lots = max(1, len(ma_signals))

    pos = 0
    positions = []
    for e in ev:
        if e > 0:
            pos = min(pos + e, max_lots)       # 加仓 e 手
        elif e < 0:
            pos = max(pos // 2, 0)             # 减半（向下取整）
        # e == 0 → 保持
        positions.append(pos)

    d["MA_Position"] = positions
    d["MA_Weight"]   = d["MA_Position"] / float(max_lots)
    
    return d


def backtest_ma_strategy(df, price="Close", weight_col="MA_Weight",
                         fee_bps=5, freq=252, rf_annual=0.0):
    """
    Minimal backtest:
      - uses df[weight_col] in [0,1]
      - daily net return = weight(t-1) * pct_change(price) - |Δweight| * fee_rate
      - returns sharpe, cagr, mdd, daily_return (Series), equity (Series)
    """
    s = pd.to_numeric(df[price], errors="coerce")
    w = pd.to_numeric(df[weight_col], errors="coerce").fillna(0.0).clip(0.0, 1.0)

    ret_u = s.pct_change().fillna(0.0)
    ret   = w.shift(1).fillna(0.0) * ret_u

    turnover = w.diff().abs().fillna(w.abs())
    ret -= turnover * (fee_bps / 10000.0)

    equity = (1 + ret).cumprod()

    n = len(ret)
    mdd = float((equity / equity.cummax() - 1.0).min())
    cagr = float(equity.iloc[-1]**(freq / max(n,1)) - 1.0) if n > 0 else 0.0

    rf_daily = rf_annual / freq
    vol = float(ret.std(ddof=0))
    sharpe = float(np.sqrt(freq) * (ret.mean() - rf_daily) / vol) if vol > 0 else 0.0

    return sharpe, cagr, mdd


            


def bb_crossover(df,
    price="Close",
    lower_1="Lower_Band_20", lower_2="Lower_Band_60",
    mid_col="MA_20", upper20_col="Upper_Band_20",
    ttl=None,              # 跌破20下轨后最大等待天数；None=不限
    layer_cap=2,           # 最大持仓层数（允许叠加）
    enter_layers=1,        # 每次入场加几层
    layers_mid=1           # 触及中轨先减几层；剩余到上轨再减完
):
    d = df.copy()
    d["BB_signal"] = 0

    armed = False
    wait_left = None
    pos_layers = 0         # 当前持仓层数
    mid_taken = False      # 本轮持仓是否已在中轨减过仓（防重复）

    for i in range(len(d)):
        p  = d[price].iat[i]
        l1 = d[lower_1].iat[i]
        l2 = d[lower_2].iat[i]
        m  = d[mid_col].iat[i]
        u1 = d[upper20_col].iat[i]

        if np.isnan(p) or np.isnan(l1) or np.isnan(l2) or np.isnan(m) or np.isnan(u1):
            continue

        # first time below lower20 - wait
        if (not armed) and pos_layers == 0 and p < l1:
            armed = True
            wait_left = (None if ttl is None else int(ttl))

        # second time crossover lower20 or below lower60 - buy
        if armed and pos_layers == 0:
            if p <= l2 or p >= l1:
                add = min(enter_layers, layer_cap - pos_layers)
                if add > 0:
                    d["BB_signal"].iat[i] = +add
                    pos_layers += add
                    armed = False
                    wait_left = None
                    mid_taken = False
            else:
                if ttl is not None:
                    wait_left -= 1
                    if wait_left <= 0:
                        armed = False
                        wait_left = None

        # take profit in 2 steps
        if pos_layers > 0:
            # take profit at mid band
            if (not mid_taken) and p >= m:
                x = min(layers_mid, pos_layers)
                d["BB_signal"].iat[i] = -x
                pos_layers -= x
                mid_taken = True if pos_layers > 0 else False
                if pos_layers == 0:
                    armed = False
                    wait_left = None
                continue

            # take profit at upper band
            if p >= u1:
                d["BB_signal"].iat[i] = -pos_layers
                pos_layers = 0
                armed = False
                wait_left = None
                mid_taken = False
                continue

    return d


def bb_strategy(df,
    price="Close",
    lower_1="Lower_Band_20", lower_2="Lower_Band_60",
    mid_col="MA_20", upper20_col="Upper_Band_20",
    ttl=None, layer_cap=2, enter_layers=1, layers_mid=1,
    date_col="date"
):
    # 排序（若有日期列）
    d = df.sort_values(date_col).copy() if date_col in df.columns else df.copy()

    # 生成事件（+n 进、-n 出、0 无操作）
    d = bb_crossover(
        d, price=price,
        lower_1=lower_1, lower_2=lower_2,
        mid_col=mid_col, upper20_col=upper20_col,
        ttl=ttl, layer_cap=layer_cap,
        enter_layers=enter_layers, layers_mid=layers_mid
    )

    # 事件累计为持仓/权重（只做多）
    ev = pd.to_numeric(d["BB_signal"], errors="coerce").fillna(0).astype(int)
    d["BB_Position"] = ev.cumsum().clip(lower=0, upper=layer_cap).astype(int)
    d["BB_Weight"]   = d["BB_Position"] / float(layer_cap)
    return d


def backtest_bb_strategy(df, price="Close", weight_col="BB_Weight",
                         fee_bps=5, freq=252, rf_annual=0.0):
    """
    返回: sharpe, cagr, mdd
    日收益 = weight(t-1) * pct_change(price) - |Δweight| * fee_rate
    """
    s = pd.to_numeric(df[price], errors="coerce")
    w = pd.to_numeric(df[weight_col], errors="coerce").fillna(0.0).clip(0.0, 1.0)

    ret_u = s.pct_change().fillna(0.0)
    ret   = w.shift(1).fillna(0.0) * ret_u

    turnover = w.diff().abs().fillna(w.abs())
    ret -= turnover * (fee_bps / 10000.0)

    equity = (1 + ret).cumprod()
    n = len(ret)
    mdd  = float((equity / equity.cummax() - 1.0).min())
    cagr = float(equity.iloc[-1]**(freq / max(n,1)) - 1.0) if n > 0 else 0.0

    rf_daily = rf_annual / freq
    vol = float(ret.std(ddof=0))
    sharpe = float(np.sqrt(freq) * (ret.mean() - rf_daily) / vol) if vol > 0 else 0.0
    return sharpe, cagr, mdd