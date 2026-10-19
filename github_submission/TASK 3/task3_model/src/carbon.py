import hashlib
import numpy as np
import pandas as pd
from .util import DATA,read

REGIONS=['DE','FR','ES']


class Carbon:
    def __init__(self,frame,anchor,metadata=None):
        f=frame.copy()
        if 'datetime_utc' not in f or 'carbon_intensity_gco2e_kwh' not in f:raise ValueError('CO2e canonical columns required')
        f['datetime_utc']=pd.to_datetime(f.datetime_utc,utc=True,errors='raise')
        if f.duplicated(['datetime_utc','region']).any():raise ValueError('Duplicate carbon observations')
        pivot=f.pivot(index='datetime_utc',columns='region',values='carbon_intensity_gco2e_kwh').sort_index()
        if set(pivot.columns)!=set(REGIONS) or not np.isfinite(pivot.to_numpy()).all() or (pivot.to_numpy()<0).any():raise ValueError('Missing/negative carbon')
        if not pivot.index.equals(pd.date_range(pivot.index[0],pivot.index[-1],freq='h')):raise ValueError('Non-contiguous hourly carbon')
        self.values=pivot[REGIONS].to_numpy(float)
        self.origin=(pivot.index[0]-anchor).total_seconds()
        self.anchor=anchor
        self.metadata=metadata or {'data_kind':'synthetic_test_only','unit':'gCO2e/kWh','boundary':'synthetic_test_only'}

    @classmethod
    def load(cls,anchor,boundary='lifecycle',spread=1.):
        path=DATA/('carbon_'+boundary+'.csv')
        if not path.is_file():raise FileNotFoundError('Real carbon missing; no zero/synthetic fallback')
        f=pd.read_csv(path)
        if spread!=1:
            means=f.groupby('datetime_utc').carbon_intensity_gco2e_kwh.transform('mean')
            f['carbon_intensity_gco2e_kwh']=np.maximum(0,means+spread*(f.carbon_intensity_gco2e_kwh-means))
        obj=cls(f,anchor,read(DATA/('carbon_'+boundary+'.metadata.json')))
        obj.metadata={**obj.metadata,'derived_spatial_spread':spread,'hypothetical_transform':spread!=1}
        return obj

    def index(self,t):
        i=int(np.floor((t-self.origin)/3600))
        if not 0<=i<len(self.values):raise ValueError('Carbon coverage missing at '+str(t))
        return i

    def at(self,t,region):return float(self.values[self.index(t),REGIONS.index(region)])

    def integral(self,start,end,region):
        if end<start:raise ValueError('Negative carbon integration interval')
        if end==start:return 0.
        self.index(start);self.index(np.nextafter(end,-np.inf))
        value=0.;t=start
        while t<end-1e-10:
            i=self.index(t);right=min(end,self.origin+(i+1)*3600)
            value+=(right-t)*self.values[i,REGIONS.index(region)];t=right
        return float(value)

    def observation_history(self,issued):
        last=int(np.floor((issued-self.origin)/3600))-1
        if last<0:raise ValueError('No fully observed past carbon hour; cannot initialize persistence')
        if last>=len(self.values):raise ValueError('Observation coverage unavailable')
        return self.values[:last+1].copy(),self.origin+(last+1)*3600


class ForecastView:
    kind='causal_forecast'
    def __init__(self,actual,issued,horizon,method='persistence',noise=0.,seed=20261017):
        history,cutoff=actual.observation_history(issued)
        self.issued=issued;self.cutoff=cutoff;self.end=issued+horizon+3600
        self.origin=actual.origin;self.anchor=actual.anchor
        self.table={};self.method=method
        for hour in range(int(np.floor((issued-self.origin)/3600)),int(np.ceil((self.end-self.origin)/3600))+1):
            if method=='persistence':row=history[-1]
            elif method=='historical_hour':
                part=history[np.arange(len(history))%24==hour%24]
                row=part.mean(axis=0) if len(part) else history.mean(axis=0)
            else:raise ValueError('Unknown causal forecast method')
            for r,value in zip(REGIONS,row):
                if noise:
                    token=f'{seed}|{issued:.6f}|{hour}|{r}'.encode()
                    local=int(hashlib.sha256(token).hexdigest()[:16],16)
                    value*=np.exp(np.random.default_rng(local).normal(-.5*noise**2,noise))
                self.table[(hour,r)]=float(value)
        if cutoff>issued+1e-8:raise AssertionError('Future observation used')

    def at(self,t,r):
        if t<self.issued-1e-6 or t>self.end+1e-8:raise ValueError('Forecast outside issued horizon')
        return self.table[(int(np.floor((t-self.origin)/3600)),r)]

    def integral(self,start,end,r):
        total=0.;t=start
        while t<end-1e-10:
            hour=int(np.floor((t-self.origin)/3600));right=min(end,self.origin+(hour+1)*3600)
            total+=(right-t)*self.at(t,r);t=right
        return total

    def records(self):
        return [{'issued_at_utc':(self.anchor+pd.Timedelta(seconds=self.issued)).isoformat(),
            'datetime_utc':(self.anchor+pd.Timedelta(seconds=self.origin+hour*3600)).isoformat(),'region':r,
            'predicted_carbon_intensity_gco2_kwh':value,'emissions_species':'CO2e','unit':'gCO2e/kWh',
            'observation_end_utc':(self.anchor+pd.Timedelta(seconds=self.cutoff)).isoformat(),
            'forecast_method':self.method} for (hour,r),value in self.table.items()]


class OracleView:
    kind='perfect_foresight_oracle'
    def __init__(self,actual):self._actual=actual;self.origin=actual.origin
    def at(self,t,r):return self._actual.at(t,r)
    def integral(self,a,b,r):return self._actual.integral(a,b,r)
