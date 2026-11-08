#!/usr/bin/env python3
"""Deribit raw-preserving backfill and C/D data-gate audit.

Run: python code/02_deribit.py --root . --stage all
Replay: python code/02_deribit.py --root . --stage all --offline
Only raw API candles and actual historical option trades are inputs. Minute
timestamps are candle starts, with candle closes public at timestamp+60s.
"""
from __future__ import annotations
import argparse, concurrent.futures, hashlib, json, math, os, threading, time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import numpy as np
import pandas as pd

SEED = 20261007
START = "2021-03-01"
END = "2026-10-07"  # Exclusive: only completed UTC days through 2026-10-06.
PILOT_DATES = ["2021-03-25", "2021-09-08", "2023-01-11", "2023-06-15",
               "2025-01-16", "2025-08-19", "2026-01-14", "2026-10-06"]
LOCK = threading.Lock()

def ms(s):
    return int(pd.Timestamp(s, tz="UTC").timestamp() * 1000)

def iso(t):
    return pd.Timestamp(t, unit="ms", tz="UTC").isoformat()

def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()

def log(root, text):
    with LOCK:
        with (root / "logs/deribit_run.md").open("a") as f:
            f.write(f"- {datetime.now(timezone.utc).isoformat()}：{text}\n")
        print(text, flush=True)

class API:
    def __init__(self, root, offline=False):
        self.root, self.offline = root, offline
        self.raw = root / "data_raw/deribit"
        if offline:
            if not self.raw.is_dir():raise FileNotFoundError(f"Offline raw directory missing: {self.raw}")
        else:self.raw.mkdir(parents=True, exist_ok=True)
        self.receipts = root / "logs/deribit_requests.jsonl"

    def get(self, domain, method, params, group):
        canonical = json.dumps({"domain": domain, "method": method,
                                "params": params}, sort_keys=True)
        key = hashlib.sha256(canonical.encode()).hexdigest()[:20]
        dest = self.raw / group / f"{method}_{params.get('currency','all')}_{key}.json"
        if self.offline:
            if not dest.parent.is_dir():raise FileNotFoundError(f"Offline raw group missing: {dest.parent}")
        else:dest.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://{domain}/api/v2/public/{method}?" + urlencode(params)
        if dest.exists():
            try:
                obj = json.loads(dest.read_text())
                if "result" in obj and "error" not in obj:
                    return obj["result"]
            except (json.JSONDecodeError, OSError):
                if self.offline:
                    raise
        if self.offline:
            raise FileNotFoundError(f"Offline raw response missing: {dest}")
        for attempt in range(5):
            started = datetime.now(timezone.utc).isoformat()
            status, body, error = None, b"", None
            try:
                req = Request(url, headers={"User-Agent": "JOD reproducible academic research/1.0"})
                with urlopen(req, timeout=35) as r:
                    status, body = r.status, r.read()
                obj = json.loads(body)
                if "error" in obj:
                    error = json.dumps(obj["error"])
                elif "result" not in obj:
                    error = "JSON response has no result"
            except HTTPError as e:
                status, body, error = e.code, e.read(), str(e)
            except Exception as e:
                error = repr(e)
            # Preserve each actual request response, including errors.
            attempt_path = dest.with_name(dest.stem + f"_attempt{attempt}_{time.time_ns()}.response")
            attempt_path.write_bytes(body)
            receipt = dict(started_at=started, url=url, params=params, http_status=status,
                           error=error, attempt=attempt,
                           response=str(attempt_path.relative_to(self.root)),
                           sha256=digest(attempt_path), bytes=len(body))
            with LOCK:
                with self.receipts.open("a") as f:
                    f.write(json.dumps(receipt, ensure_ascii=False) + "\n")
            if error is None:
                tmp = dest.with_suffix(".json.part")
                tmp.write_bytes(body)
                tmp.replace(dest)
                time.sleep(0.06)
                return obj["result"]
            if status not in (None, 429, 500, 502, 503, 504) and "too_many" not in str(error):
                break
            time.sleep(min(16, 2 ** attempt))
        raise RuntimeError(f"{domain}/{method}: {error}")

    def dvol(self, currency, resolution):
        start, end = ms(START), ms(END) - (60000 if resolution == "60" else 1)
        blocks, pages, empty_pages = [], 0, 0
        seen_end = set()
        while end >= start:
            if end in seen_end:
                raise RuntimeError("Repeated DVOL continuation")
            seen_end.add(end)
            params = dict(currency=currency, start_timestamp=start,
                          end_timestamp=end, resolution=resolution)
            result = self.get("www.deribit.com", "get_volatility_index_data",
                              params, f"dvol/{resolution}/{currency}")
            data = result.get("data", [])
            blocks.extend(data)
            pages += 1
            if not data:
                empty_pages += 1
            continuation = result.get("continuation")
            if pages % 25 == 0:
                log(self.root, f"DVOL {currency}/{resolution}：{pages} 页，{len(blocks)} 行，游标 {iso(end)}。")
            if continuation is None:
                break
            if int(continuation) >= end:
                raise RuntimeError("DVOL continuation fails to decrease")
            end = int(continuation)  # Official continuation is next end_timestamp.
        df = pd.DataFrame(blocks, columns=["timestamp", "open", "high", "low", "close"])
        if len(df):
            df = df.drop_duplicates("timestamp").sort_values("timestamp")
            df = df[(df.timestamp >= start) & (df.timestamp < ms(END))]
            df["datetime"] = pd.to_datetime(df.timestamp, unit="ms", utc=True)
            step = 60000 if resolution == "60" else 3600000 if resolution == "3600" else 86400000
            df["public_at"] = pd.to_datetime(df.timestamp + step, unit="ms", utc=True)
        df["currency"], df["asset"] = currency, currency + "USDT"
        df.to_parquet(self.root / f"data_clean/dvol_{'minute' if resolution=='60' else 'hourly' if resolution=='3600' else 'reference_daily'}_{currency}.parquet", index=False)
        log(self.root, f"DVOL {currency}/{resolution} 完成：{pages} 页，{len(df)} 行，空页 {empty_pages}，起点 {iso(df.timestamp.min()) if len(df) else '空'}。")
        return df

    def option_day(self, currency, day):
        # Five real pre-origin minutes seed the UTC opening grid. They are
        # excluded from today's trade-count denominator, never future backfill.
        start, end, cursor = ms(day), ms(day) + 86400000 - 1, ms(day)-300000
        seen, rows, pages = set(), [], 0
        while cursor <= end:
            params = dict(currency=currency, kind="option", start_timestamp=cursor,
                          end_timestamp=end, count=1000, sorting="asc")
            result = self.get("history.deribit.com", "get_last_trades_by_currency_and_time",
                              params, f"options/{currency}/{day}")
            trades = result.get("trades", [])
            fresh = [r for r in trades if r.get("trade_id") not in seen]
            for r in fresh:
                seen.add(r.get("trade_id"))
            rows.extend(fresh)
            pages += 1
            if not trades or not result.get("has_more", False):
                break
            next_cursor = max(int(r["timestamp"]) for r in trades)
            if next_cursor < cursor or (next_cursor == cursor and not fresh):
                raise RuntimeError(f"Timestamp pagination stalled for {currency} {day}")
            cursor = next_cursor  # Inclusive overlap retains simultaneous boundary trades.
            if pages % 25 == 0:
                log(self.root, f"期权 {currency}/{day}：{pages} 页，{len(rows)} 条真实成交。")
        df = pd.DataFrame(rows)
        df["currency"], df["date"] = currency, day
        if len(df):
            df = df.drop_duplicates("trade_id").sort_values(["timestamp", "trade_id"])
        df.to_parquet(self.root / f"data_clean/option_trades_{currency}_{day}.parquet", index=False)
        log(self.root, f"期权 {currency}/{day} 全日分页完成：{pages} 页，{len(df)} 条成交。")
        return df

    def instruments(self, currency):
        # Existing preliminary probe is the same official complete response.
        probe = self.raw / "probes" / f"history.deribit.com_get_instruments_{currency}_expired_.json"
        if probe.exists():
            obj = json.loads(probe.read_text())
            attrs = obj.get("result", [])
            provenance = {"source": str(probe.relative_to(self.root)), "sha256": digest(probe),
                          "url": f"https://history.deribit.com/api/v2/public/get_instruments?currency={currency}&kind=option&expired=true"}
            (self.root / f"out/D/instruments_receipt_{currency}.json").write_text(json.dumps(provenance, indent=2))
        else:
            attrs = self.get("history.deribit.com", "get_instruments",
                             dict(currency=currency, kind="option", expired="true"), "instruments")
        # The latest audit day also trades contracts that have not expired yet.
        # Combine their actual production metadata; creation timestamps still
        # govern whether a contract belonged to the historical origin universe.
        active = self.get("www.deribit.com", "get_instruments",
                          dict(currency=currency, kind="option", expired="false"), "instruments")
        expired_count = len(attrs)
        attrs = attrs + active
        df = pd.DataFrame(attrs)
        needed = ["instrument_name", "expiration_timestamp", "creation_timestamp", "strike",
                  "option_type", "contract_size", "quote_currency", "settlement_currency",
                  "maker_commission", "taker_commission"]
        df = df[[x for x in needed if x in df]].drop_duplicates("instrument_name")
        df["currency"] = currency
        df.attrs["expired_metadata_count"] = expired_count
        df.attrs["active_metadata_count"] = len(active)
        df.to_parquet(self.root / f"data_clean/option_instruments_{currency}.parquet", index=False)
        return df

def dvol_daily(df):
    if df.empty:
        return pd.DataFrame()
    d = df.copy().sort_values("timestamp")
    d["date"] = d.datetime.dt.floor("D")
    valid = (d.close > 0) & np.isfinite(d.close)
    # Close known at timestamp+1m; adjacent-minute differences only, never gaps.
    r = np.log(d.close.where(valid)).diff().where(d.timestamp.diff().eq(60000))
    hist_scale = r.shift(1).abs().rolling(60, min_periods=30).median() / 0.67448975
    jump = r.abs() > 4.5 * hist_scale
    d["change"], d["jump_local"] = r, jump
    rows = []
    theta = (math.pi / 2) ** 2 + math.pi - 5
    mu43 = 2 ** (2/3) * math.gamma(7/6) / math.sqrt(math.pi)
    for date, g in d.groupby("date"):
        x = g.change.to_numpy(float)
        finite = np.isfinite(x)
        xx = x[finite]
        rv = np.nansum(x*x)
        product = np.abs(x[1:] * x[:-1])
        n = len(x)
        bv = math.pi/2 * n/max(n-1,1) * np.nansum(product)
        trip = np.abs(x[2:] * x[1:-1] * x[:-2]) ** (4/3)
        tpq = n*n/max(n-2,1) * np.nansum(trip) / mu43**3
        z = ((rv-bv)/rv) / math.sqrt(max(theta/max(n,1)*max(1,tpq/max(bv*bv,1e-30)),1e-30)) if rv > 0 else 0
        local = g.jump_local.fillna(False).to_numpy(bool)
        last = g.iloc[-1]
        rows.append(dict(date=date, currency=last.currency, asset=last.asset,
                         minute_count=len(g), expected_minutes=1440,
                         minute_missing_rate=1-len(g)/1440,
                         valid_changes=int(finite.sum()),
                         iv_level_last=float(last.close), iv_variance_daily=(float(last.close)/100)**2/365,
                         iv_rv=rv, iv_rs_positive=np.sum(xx[xx>=0]**2),
                         iv_rs_negative=np.sum(xx[xx<0]**2), iv_bpv=bv,
                         iv_bns_stat=z, iv_bns_jump=max(rv-bv,0) if z>2.326347874 else 0,
                         iv_jump_count=int(local.sum()), iv_jump_amplitude=np.nansum(x[local]**2),
                         iv_last_public_at=last.public_at,
                         complete_day=len(g)>=1433))
    return pd.DataFrame(rows)

def audit_options(df, attrs, currency, day):
    """Real contract-minute denominator and <=5min IV interpolation audit."""
    start, end = ms(day), ms(day)+86400000
    if df.empty:
        return [], [], dict(currency=currency,date=day,trades=0,reason="empty complete API day")
    x = df.merge(attrs, on="instrument_name", how="left", validate="many_to_one", suffixes=("", "_attrs"))
    actual_today = x[x.timestamp >= start]
    missing = [dict(currency=currency,date=day,field=f,rows=len(actual_today),
                    nonmissing=int(pd.to_numeric(actual_today[f],errors="coerce").notna().sum()) if f in actual_today else 0,
                    coverage=float(pd.to_numeric(actual_today[f],errors="coerce").notna().mean()) if f in actual_today else 0)
               for f in ["price", "iv", "index_price", "strike", "expiration_timestamp", "contract_size", "bid_price", "ask_price"]]
    valid = x.price.gt(0) & x.iv.gt(0) & x.index_price.gt(0) & x.strike.gt(0)
    valid &= x.expiration_timestamp.notna()
    actual_price_iv_index_coverage = float(valid[x.timestamp >= start].mean())
    x = x[valid].copy()
    # Fixed day's initial observed reference, strictly public at/after day opening.
    before = x[x.timestamp < start].sort_values("timestamp")
    if not len(before):
        raise RuntimeError(f"No real index reference before UTC origin: {currency}/{day}")
    reference = float(before.iloc[-1].index_price)
    a = attrs[(attrs.expiration_timestamp >= start) & (attrs.expiration_timestamp <= end+30*86400000)
              & (attrs.creation_timestamp <= start)].copy()
    grid = np.arange(start, end, 60000, dtype=np.int64)
    index_t = x.sort_values("timestamp").timestamp.to_numpy(np.int64)
    index_j = np.searchsorted(index_t,grid,side="right")-1
    index_ok = (index_j>=0) & ((grid-index_t[np.maximum(index_j,0)])<=300000)
    grids, details = {}, []
    for name, group in x.groupby("instrument_name"):
        group = group.sort_values("timestamp")
        tt = group.timestamp.to_numpy(np.int64)
        j = np.searchsorted(tt,grid,side="right")-1
        ok = (j>=0) & ((grid-tt[np.maximum(j,0)])<=300000)
        v = np.full(1440,np.nan)
        v[ok] = group.iv.to_numpy(float)[j[ok]]
        grids[name] = v
    for (expiry, kind), group in a.groupby(["expiration_timestamp", "option_type"]):
        # Choose 0D separately and 1-30D, no metadata-created-before-listing backfill.
        tenor = (expiry-start)/86400000
        if tenor > 30 or tenor < 0:
            continue
        group = group.sort_values("strike").drop_duplicates("strike")
        strikes = group.strike.to_numpy(float)
        names = group.instrument_name.to_numpy(str)
        if not len(strikes):
            continue
        targets = np.unique([int(np.abs(strikes/reference-k).argmin()) for k in [.90,.95,1,1.05,1.10]])
        values = np.vstack([grids.get(n,np.full(1440,np.nan)) for n in names])
        for j in targets:
            meta = group.iloc[j]
            alive = (grid >= max(start, int(meta.creation_timestamp))) & (grid < min(end,int(expiry)))
            denominator = int(alive.sum())
            if not denominator:
                continue
            direct = np.isfinite(values[j]) & index_ok
            left = np.isfinite(values[:j]).any(axis=0) if j>0 else np.zeros(1440,bool)
            right = np.isfinite(values[j+1:]).any(axis=0) if j+1<len(values) else np.zeros(1440,bool)
            interpolated = (~direct) & left & right & index_ok
            reconstructed = (direct | interpolated) & alive
            rate = 1-float(reconstructed.sum())/denominator
            details.append(dict(currency=currency,date=day,instrument_name=meta.instrument_name,
                                expiration_timestamp=int(expiry),tenor_days=tenor,strike=float(meta.strike),
                                initial_index=reference,option_type=kind,expected_minutes=denominator,
                                direct_minutes=int((direct&alive).sum()),interpolated_minutes=int((interpolated&alive).sum()),
                                reconstructed_minutes=int(reconstructed.sum()),minute_missing_rate=rate,
                                qualified_contract_day=rate<.005))
    details_df = pd.DataFrame(details)
    total = int(details_df.expected_minutes.sum()) if len(details_df) else 0
    reconstructed = int(details_df.reconstructed_minutes.sum()) if len(details_df) else 0
    today = df[df.timestamp >= start]
    summary = dict(currency=currency,date=day,trades=len(today),actual_traded_contracts=today.instrument_name.nunique(),
                   evaluated_contract_days=len(details_df),expected_minutes=total,
                   reconstructed_minutes=reconstructed,minute_missing_rate=1-reconstructed/total if total else np.nan,
                   qualified_contract_days=int(details_df.qualified_contract_day.sum()) if len(details_df) else 0,
                   complete_day=total>0 and (1-reconstructed/total)<.005,
                   actual_price_iv_index_coverage=actual_price_iv_index_coverage)
    return details, missing, summary

def run_dvol(root, api):
    daily, gates = [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        minute = dict(zip(["BTC","ETH"], pool.map(lambda c: api.dvol(c,"60"),["BTC","ETH"])))
    for currency, df in minute.items():
        d = dvol_daily(df)
        daily.append(d)
        if len(df):
            span = int((df.timestamp.max()-df.timestamp.min())//60000)+1
            within_missing = 1-len(df)/span
            requested = (ms(END)-ms(START))//60000
            complete = int(d.complete_day.sum())
            coverage = float(d.loc[d.complete_day,["iv_rv","iv_level_last","iv_rs_positive","iv_rs_negative","iv_jump_count","iv_jump_amplitude"]].notna().all(axis=1).mean()) if complete else 0
            gates.append(dict(currency=currency,start=iso(df.timestamp.min()),end=iso(df.timestamp.max()),days=len(d),
                              complete_days=complete,observed_minutes=len(df),expected_minutes_within_observed_span=span,
                              minute_missing_rate=within_missing,requested_expected_minutes=requested,
                              requested_minute_missing_rate=1-len(df)/requested,variable_coverage=coverage,
                              gate1_pass=complete>=1000 and within_missing<.005 and coverage>=.95,
                              reason="complete minute days below 1000" if complete<1000 else "threshold check"))
        else:
            gates.append(dict(currency=currency,days=0,complete_days=0,observed_minutes=0,minute_missing_rate=1,
                              variable_coverage=0,gate1_pass=False,reason="minute API returned no data"))
    pd.concat(daily,ignore_index=True).to_parquet(root/"data_clean/dvol_daily.parquet",index=False)
    pd.DataFrame(gates).to_csv(root/"out/C/data_gate.csv",index=False)
    # Collect complete hourly history as a provenance/retention comparison only.
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda c: api.dvol(c,"3600"),["BTC","ETH"]))
    plot_dvol(root)
    log(root,"主题丙数据闸门已写 out/C/data_gate.csv；小时参考不替代分钟变量。")

def run_options(root, api):
    attrs = {c:api.instruments(c) for c in ["BTC","ETH"]}
    availability=[]
    for c in ["BTC","ETH"]:
        for order in ["asc","desc"]:
            result=api.get("history.deribit.com","get_last_trades_by_currency_and_time",
                           dict(currency=c,kind="option",start_timestamp=ms("2016-01-01"),
                                end_timestamp=ms(END)-1,count=1,sorting=order),"availability")
            t=(result.get("trades",[]) or [{}])[0]
            availability.append(dict(currency=c,sorting=order,timestamp=t.get("timestamp"),
                                     date=iso(t["timestamp"]) if t.get("timestamp") else None,
                                     instrument=t.get("instrument_name"),has_more=result.get("has_more"),
                                     instrument_metadata_count=len(attrs[c]),
                                     expired_metadata_count=attrs[c].attrs["expired_metadata_count"],
                                     active_metadata_count=attrs[c].attrs["active_metadata_count"]))
    pd.DataFrame(availability).to_csv(root/"out/D/history_availability.csv",index=False)
    details,fields,summaries=[],[],[]
    jobs=[(c,d) for d in PILOT_DATES for c in ["BTC","ETH"]]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        results=list(pool.map(lambda j: (j,api.option_day(*j)),jobs))
    for (c,day),df in results:
        a,b,s=audit_options(df,attrs[c],c,day)
        details.extend(a);fields.extend(b);summaries.append(s)
    pd.DataFrame(details).to_csv(root/"out/D/contract_minute_coverage.csv",index=False)
    pd.DataFrame(fields).to_csv(root/"out/D/field_coverage.csv",index=False)
    summaries=pd.DataFrame(summaries)
    summaries.to_csv(root/"out/D/day_coverage.csv",index=False)
    gates=[]
    for c,g in summaries.groupby("currency"):
        denominator=int(g.expected_minutes.sum())
        rebuilt=int(g.reconstructed_minutes.sum())
        minute_missing=1-rebuilt/denominator if denominator else 1
        qualified=int(g.qualified_contract_days.sum())
        complete=int(g.complete_day.sum())
        coverage=float(np.average(g.actual_price_iv_index_coverage,weights=g.trades))
        av=[r for r in availability if r['currency']==c and r.get('timestamp')]
        span=int((max(r['timestamp'] for r in av)-min(r['timestamp'] for r in av))//86400000)+1 if av else 0
        gates.append(dict(currency=c,start=g.date.min(),end=g.date.max(),days=complete,
                          audited_calendar_days=len(g),source_history_span_days=span,
                          minute_missing_rate=minute_missing,variable_coverage=coverage,
                          contract_days=qualified,evaluated_contract_days=int(g.evaluated_contract_days.sum()),
                          expected_contract_minutes=denominator,reconstructed_contract_minutes=rebuilt,
                          actual_trade_count=int(g.trades.sum()),
                          gate1_pass=complete>=1000 and minute_missing<.005 and coverage>=.95,
                          reason="predefined cross-year minute reconstruction audit fails 0.5% missing threshold" if minute_missing>=.005 else "pilot passes; full-history collection required",
                          scope="8 predefined cross-year complete API trade days; undownloaded history excluded"))
    pd.DataFrame(gates).to_csv(root/"out/D/data_gate.csv",index=False)
    plot_options(root)
    log(root,"主题丁实际分钟网格覆盖、字段覆盖和数据闸门已写 out/D/；未下载历史不进入缺失分母。")

def plotting(root):
    os.environ["MPLCONFIGDIR"]=str(root/"logs/matplotlib_cache")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,
                         "axes.spines.top":False,"axes.spines.right":False})
    return plt

def plot_dvol(root):
    plt=plotting(root);d=pd.read_parquet(root/"data_clean/dvol_daily.parquet")
    fig,ax=plt.subplots(figsize=(8,3.6))
    for c,g in d.groupby("currency"):
        ax.plot(g.date,g.iv_level_last,label=c,lw=1)
    ax.set(ylabel="DVOL close (annual volatility, %)",xlabel="UTC date")
    ax.legend(frameon=False);fig.tight_layout()
    fig.savefig(root/"out/C/observed_minute_dvol_levels.png",dpi=300);plt.close(fig)
    rates=[]
    for c in ["BTC","ETH"]:
        hour=pd.read_parquet(root/f"data_clean/dvol_hourly_{c}.parquet")
        rates.append(dict(currency=c,minute_days=int(d[d.currency==c].complete_day.sum()),
                          hourly_calendar_days=hour.datetime.dt.floor("D").nunique(),
                          hourly_rows=len(hour)))
    table=pd.DataFrame(rates);table.to_csv(root/"out/C/resolution_coverage.csv",index=False)
    fig,ax=plt.subplots(figsize=(6,4.2));x=np.arange(2)
    ax.bar(x-.18,table.minute_days,width=.36,label="Complete minute days")
    ax.bar(x+.18,table.hourly_calendar_days,width=.36,label="Hourly reference days")
    ax.axhline(1000,c="black",ls="--",lw=1,label="Required minute days: 1000")
    ax.set_xticks(x,["BTC","ETH"]);ax.set(ylabel="UTC days")
    ax.legend(frameon=False,bbox_to_anchor=(0.5,1.28),loc="upper center",fontsize=8);fig.tight_layout()
    fig.savefig(root/"out/C/dvol_resolution_coverage.png",dpi=300);plt.close(fig)

def plot_options(root):
    plt=plotting(root);d=pd.read_csv(root/"out/D/day_coverage.csv")
    fig,ax=plt.subplots(figsize=(9,3.8));dates=list(PILOT_DATES);x=np.arange(len(dates))
    for j,c in enumerate(["BTC","ETH"]):
        g=d[d.currency==c].set_index("date").reindex(dates)
        ax.bar(x+(j-.5)*.36,100*(1-g.minute_missing_rate),width=.36,label=c)
    ax.axhline(99.5,c="black",ls="--",lw=1,label="Required: 99.5%")
    ax.set_xticks(x,dates,rotation=35,ha="right")
    ax.set(ylabel="Contract-minute coverage (%)",ylim=(0,105))
    ax.legend(frameon=False);fig.tight_layout()
    fig.savefig(root/"out/D/option_minute_coverage_by_date.png",dpi=300);plt.close(fig)
    c=pd.read_csv(root/"out/D/contract_minute_coverage.csv")
    c["tenor_group"]=pd.cut(c.tenor_days,[0,1,3,7,14,30],labels=["0D","1-3D","4-7D","8-14D","15-30D"],include_lowest=True)
    rows=[]
    for (cur,tenor),g in c.groupby(["currency","tenor_group"],observed=True):
        rows.append(dict(currency=cur,tenor_group=str(tenor),contract_days=len(g),
                         expected_minutes=int(g.expected_minutes.sum()),
                         reconstructed_minutes=int(g.reconstructed_minutes.sum()),
                         minute_coverage=float(g.reconstructed_minutes.sum()/g.expected_minutes.sum())))
    t=pd.DataFrame(rows);t.to_csv(root/"out/D/tenor_coverage.csv",index=False)
    fig,ax=plt.subplots(figsize=(7,3.6));groups=["0D","1-3D","4-7D","8-14D","15-30D"];x=np.arange(5)
    for j,cur in enumerate(["BTC","ETH"]):
        g=t[t.currency==cur].set_index("tenor_group").reindex(groups)
        ax.bar(x+(j-.5)*.36,100*g.minute_coverage,width=.36,label=cur)
    ax.axhline(99.5,c="black",ls="--",lw=1,label="Required: 99.5%")
    ax.set_xticks(x,groups);ax.set(ylabel="Contract-minute coverage (%)",ylim=(0,105))
    ax.legend(frameon=False);fig.tight_layout()
    fig.savefig(root/"out/D/option_minute_coverage_by_tenor.png",dpi=300);plt.close(fig)

def verify(root, before):
    checks=[]
    def check(name,passed,value):
        checks.append(dict(check=name,passed=bool(passed),value=str(value)))
    if (root/"out/C/data_gate.csv").exists():
        g=pd.read_csv(root/"out/C/data_gate.csv")
        daily=pd.read_parquet(root/"data_clean/dvol_daily.parquet")
        for _,r in g.iterrows():
            m=pd.read_parquet(root/f"data_clean/dvol_minute_{r.currency}.parquet")
            d=daily[daily.currency==r.currency]
            check(f"C_{r.currency}_unique_minutes",not m.timestamp.duplicated().any(),len(m))
            check(f"C_{r.currency}_day_sum",int(d.minute_count.sum())==len(m),int(d.minute_count.sum()))
            check(f"C_{r.currency}_gate_complete_days",int(d.complete_day.sum())==r.complete_days,r.complete_days)
            causal=(d.iv_last_public_at<=d.date+pd.Timedelta(days=1)).all()
            check(f"C_{r.currency}_public_before_forecast_origin",causal,len(d))
    if (root/"out/D/data_gate.csv").exists():
        gates=pd.read_csv(root/"out/D/data_gate.csv")
        days=pd.read_csv(root/"out/D/day_coverage.csv")
        details=pd.read_csv(root/"out/D/contract_minute_coverage.csv")
        for _,r in gates.iterrows():
            d=days[days.currency==r.currency];c=details[details.currency==r.currency]
            check(f"D_{r.currency}_minute_denominator",int(c.expected_minutes.sum())==r.expected_contract_minutes,r.expected_contract_minutes)
            check(f"D_{r.currency}_reconstructed_sum",int(c.reconstructed_minutes.sum())==r.reconstructed_contract_minutes,r.reconstructed_contract_minutes)
            check(f"D_{r.currency}_direct_plus_interpolation",(c.direct_minutes+c.interpolated_minutes==c.reconstructed_minutes).all(),len(c))
            expected_rate=1-r.reconstructed_contract_minutes/r.expected_contract_minutes
            check(f"D_{r.currency}_reported_missing",abs(expected_rate-r.minute_missing_rate)<1e-12,r.minute_missing_rate)
            check(f"D_{r.currency}_contract_days",int(c.qualified_contract_day.sum())==r.contract_days,r.contract_days)
            check(f"D_{r.currency}_actual_trade_count",int(d.trades.sum())==r.actual_trade_count,r.actual_trade_count)
            for date in PILOT_DATES:
                trades=pd.read_parquet(root/f"data_clean/option_trades_{r.currency}_{date}.parquet")
                check(f"D_{r.currency}_{date}_unique_trade_ids",not trades.trade_id.duplicated().any(),len(trades))
    ledger=root/"logs/deribit_requests.jsonl"
    if ledger.exists():
        receipts=[json.loads(s) for s in ledger.read_text().splitlines() if s.strip()]
        failures=[]
        for i,r in enumerate(receipts):
            f=root/r["response"]
            if not f.exists() or digest(f)!=r["sha256"]:
                failures.append(i+1)
        check("raw_response_receipt_hashes",not failures,f"{len(receipts)} receipts; mismatches={failures}")
    pd.DataFrame(checks).to_csv(root/"out/D/deribit_checks.csv",index=False)
    if before:
        replay=[dict(file=str(f.relative_to(root)),before_sha256=h,after_sha256=digest(f),match=h==digest(f)) for f,h in before.items()]
        pd.DataFrame(replay).to_csv(root/"out/D/replay_checks.csv",index=False)
        check("offline_replay_values",all(r["match"] for r in replay),len(replay))
        pd.DataFrame(checks).to_csv(root/"out/D/deribit_checks.csv",index=False)
    assert all(c["passed"] for c in checks),[c for c in checks if not c["passed"]]
    log(root,f"Deribit 数据一致性核对 {len(checks)} 项 PASS。")

def cleaning_receipts(root):
    rows=[]
    for cur in ["BTC","ETH"]:
        meta_path=root/f"data_clean/option_instruments_{cur}.parquet"
        if not meta_path.exists():continue
        attrs=pd.read_parquet(meta_path)
        for day in PILOT_DATES:
            f=root/f"data_clean/option_trades_{cur}_{day}.parquet"
            if not f.exists():continue
            t=pd.read_parquet(f)
            warmup=int((t.timestamp<ms(day)).sum())
            t=t[t.timestamp>=ms(day)].merge(attrs,on="instrument_name",how="left",validate="many_to_one",suffixes=("","_attrs"))
            rows.append(dict(currency=cur,date=day,rule="pre_origin_warmup_not_today_count",removed_from_daily_count=warmup,remaining=len(t)))
            for col in ["price","iv","index_price","strike","expiration_timestamp"]:
                ok=pd.to_numeric(t[col],errors="coerce").gt(0)
                removed=int((~ok).sum());t=t[ok]
                rows.append(dict(currency=cur,date=day,rule=f"positive_known_{col}",removed_from_daily_count=removed,remaining=len(t)))
    if rows:
        pd.DataFrame(rows).to_csv(root/"out/D/cleaning_counts.csv",index=False)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument("--offline",action="store_true")
    p.add_argument("--stage",choices=["all","dvol","options"],default="all")
    args=p.parse_args();root=args.root.resolve()
    if args.offline:
        if not (root/"data_raw/deribit").is_dir():raise FileNotFoundError("Offline Deribit source directory missing")
    else:(root/"data_raw/deribit").mkdir(parents=True,exist_ok=True)
    for d in ["data_clean","out/C","out/D","logs"]:
        (root/d).mkdir(parents=True,exist_ok=True)
    api=API(root,args.offline)
    fingerprints=[root/"data_clean/dvol_daily.parquet"]
    fingerprints += list((root/"data_clean").glob("dvol_minute_*.parquet"))
    fingerprints += list((root/"data_clean").glob("option_trades_*.parquet"))
    fingerprints += [root/f"out/{topic}/{file}" for topic,file in [("C","data_gate.csv"),("C","resolution_coverage.csv"),("D","data_gate.csv"),("D","day_coverage.csv"),("D","contract_minute_coverage.csv"),("D","field_coverage.csv"),("D","tenor_coverage.csv")]]
    before={f:digest(f) for f in fingerprints if f.exists()} if args.offline else {}
    if args.stage in ["all","dvol"]:run_dvol(root,api)
    if args.stage in ["all","options"]:run_options(root,api)
    raw=sorted(f for f in (root/"data_raw/deribit").rglob("*.json") if not f.name.startswith("._"))
    pd.DataFrame([dict(file=str(f.relative_to(root)),bytes=f.stat().st_size,sha256=digest(f)) for f in raw]).to_csv(root/"out/D/raw_manifest.csv",index=False)
    cleaning_receipts(root)
    verify(root,before)
    log(root,f"Deribit {args.stage} 完成；原始 JSON 清单 {len(raw)} 项。种子 {SEED}。")

if __name__=="__main__":main()
