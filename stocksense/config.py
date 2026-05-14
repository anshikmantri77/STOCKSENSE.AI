import streamlit as st

CHART_TEMPLATE = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#F9FAFB", family="Inter"),
    xaxis=dict(gridcolor="#1F2937", linecolor="#374151"),
    yaxis=dict(gridcolor="#1F2937", linecolor="#374151"),
)

INDIAN_INDICES = {
    "NIFTY 50": "^NSEI",
    "SENSEX": "^BSESN",
    "NIFTY BANK": "^NSEBANK",
}

TOP_MOVERS_SAMPLE = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "SBIN.NS", "LT.NS", "ASIANPAINT.NS", "MARUTI.NS", "TITAN.NS",
]


class StockAnalyzer:
    def __init__(self):
        self.large_cap_stocks = [
            'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'HINDUNILVR.NS',
            'INFY.NS', 'KOTAKBANK.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'ITC.NS',
            'ASIANPAINT.NS', 'LT.NS', 'AXISBANK.NS', 'MARUTI.NS', 'SUNPHARMA.NS',
            'ULTRACEMCO.NS', 'TITAN.NS', 'BAJFINANCE.NS', 'NESTLEIND.NS', 'POWERGRID.NS',
            'NTPC.NS', 'TECHM.NS', 'HCLTECH.NS', 'WIPRO.NS', 'COALINDIA.NS',
            'TATAMOTORS.NS', 'BAJAJFINSV.NS', 'ONGC.NS', 'GRASIM.NS', 'JSWSTEEL.NS',
            'TATASTEEL.NS', 'HINDALCO.NS', 'ADANIPORTS.NS', 'BRITANNIA.NS', 'SHREECEM.NS',
            'DRREDDY.NS', 'DIVISLAB.NS', 'CIPLA.NS', 'EICHERMOT.NS', 'HEROMOTOCO.NS',
            'BAJAJ-AUTO.NS', 'BPCL.NS', 'IOC.NS', 'INDUSINDBK.NS', 'APOLLOHOSP.NS',
            'HDFCLIFE.NS', 'SBILIFE.NS', 'ICICIPRULI.NS', 'ADANIENT.NS', 'M&M.NS',
            'TATACONSUM.NS', 'GODREJCP.NS', 'DABUR.NS', 'MARICO.NS', 'COLPAL.NS',
            'PIDILITIND.NS', 'BERGEPAINT.NS', 'ADANIGREEN.NS', 'ADANITRANS.NS', 'LTIM.NS',
            'MINDTREE.NS', 'MPHASIS.NS', 'PERSISTENT.NS', 'COFORGE.NS', 'LTTS.NS',
            'BIOCON.NS', 'LUPIN.NS', 'TORNTPHARM.NS', 'GLAND.NS', 'LALPATHLAB.NS',
            'AUBANK.NS', 'FEDERALBNK.NS', 'BANDHANBNK.NS', 'IDFCFIRSTB.NS',
            'RBLBANK.NS', 'YESBANK.NS', 'PNB.NS', 'CANBK.NS', 'UNIONBANK.NS',
            'BANKBARODA.NS', 'INDIANB.NS', 'CENTRALBK.NS', 'IOB.NS', 'PFC.NS',
            'RECLTD.NS', 'IRFC.NS', 'SAIL.NS', 'NMDC.NS', 'VEDL.NS',
            'JINDALSTEL.NS', 'JSPL.NS', 'WELCORP.NS', 'NATIONALUM.NS', 'RATNAMANI.NS',
            'APOLLOTYRE.NS', 'MRF.NS', 'BALKRISIND.NS', 'CEAT.NS',
            'MOTHERSON.NS', 'BOSCHLTD.NS', 'EXIDEIND.NS', 'AMARAJABAT.NS', 'TVSMOTOR.NS',
            'BAJAJHLDNG.NS', 'ESCORTS.NS', 'FORCEMOT.NS', 'ASHOKLEY.NS', 'MAHINDCIE.NS',
            'CUMMINSIND.NS', 'BHARATFORG.NS', 'RAMCOCEM.NS', 'JKCEMENT.NS', 'HEIDELBERG.NS',
            'AMBUJCEM.NS', 'ACC.NS', 'INDIACEM.NS', 'DALMIA.NS', 'JKLAKSHMI.NS',
            'BATAINDIA.NS', 'RELAXO.NS', 'LIBERTY.NS', 'VBL.NS',
            'JUBLFOOD.NS', 'WESTLIFE.NS', 'DEVYANI.NS', 'ZOMATO.NS',
            'NAUKRI.NS', 'PAYTM.NS', 'POLICYBZR.NS', 'AFFLE.NS', 'ROUTE.NS',
            'INDIAMART.NS', 'JUSTDIAL.NS', 'REDINGTON.NS', 'RATEGAIN.NS', 'TATAELXSI.NS',
            'CYIENT.NS', 'KPITTECH.NS', 'ZENSAR.NS', 'SONATSOFTW.NS', 'NIITTECH.NS',
            'L&TFH.NS', 'CHOLAFIN.NS', 'MANAPPURAM.NS', 'MUTHOOTFIN.NS',
            'GMRINFRA.NS', 'ADANIPOWER.NS', 'TATAPOWER.NS',
            'TORNTPOWER.NS', 'CESC.NS', 'JINDALSAW.NS', 'WELSPUNIND.NS', 'TRIDENT.NS',
            'PAGEIND.NS', 'HAVELLS.NS', 'VOLTAS.NS',
            'BLUESTARCO.NS', 'WHIRLPOOL.NS', 'CROMPTON.NS', 'VGUARD.NS',
            'KEI.NS', 'POLYCAB.NS', 'FINOLEX.NS', 'SIEMENS.NS', 'ABB.NS',
            'SCHNEIDER.NS', 'HONAUT.NS', 'THERMAX.NS', 'BHEL.NS', 'BEML.NS',
            'BEL.NS', 'HAL.NS', 'COCHINSHIP.NS',
            'CDSL.NS', 'BSE.NS', 'MCX.NS',
            'SUNTV.NS', 'BALRAMCHIN.NS',
            'DMART.NS', 'MCDOWELL-N.NS',
            'GODREJPROP.NS', 'OBEROIRLTY.NS', 'DLF.NS', 'PRESTIGE.NS',
            'BRIGADE.NS', 'SOBHA.NS', 'PHOENIXLTD.NS', 'PVRINOX.NS', 'CONCOR.NS',
            'FORTIS.NS', 'MAXHEALTH.NS', 'NHPC.NS', 'SJVN.NS',
            'HINDPETRO.NS', 'MRPL.NS', 'GAIL.NS',
            'PETRONET.NS', 'IGL.NS', 'MGL.NS', 'GSPL.NS',
        ]

        self.mid_cap_stocks = [
            'JSWENERGY.NS', 'RENUKA.NS', 'DHAMPUR.NS', 'BAJAJCON.NS', 'EMAMILTD.NS',
            'GODREJIND.NS', 'JYOTHYLAB.NS', 'CHOLAHLDNG.NS', 'TIMKEN.NS', 'SKFINDIA.NS',
            'SCHAEFFLER.NS', 'NRB.NS', 'FINEORG.NS', 'SUPRAJIT.NS', 'ENDURANCE.NS',
            'SUNDRMFAST.NS', 'MINDAIND.NS', 'SWARAJENG.NS', 'KIOCL.NS', 'SHRIRAMFIN.NS',
            'SRTRANSFIN.NS', 'CAPLIPOINT.NS', 'ESAFSFB.NS', 'SURYODAY.NS', 'FINPIPE.NS',
            'CAMS.NS', 'CARERATING.NS', 'ICRA.NS',
            'MOTILALOF.NS', 'ANGELONE.NS', 'IIFL.NS', 'GEOJITFSL.NS', 'VENKEYS.NS',
            'KRBL.NS', 'TEXRAIL.NS',
            'KNRCON.NS', 'IRB.NS', 'GPPL.NS',
            'KANSAINER.NS', 'AIAENG.NS', 'KIRLOSENG.NS', 'CARYSIL.NS',
            'DIXON.NS', 'AMBER.NS', 'MINDA.NS', 'SUNDARAM.NS', 'GLENMARK.NS',
            'CADILAHC.NS', 'ALKEM.NS', 'AJANTPHARM.NS', 'ABBOTINDIA.NS', 'PFIZER.NS',
            'NOVARTIS.NS', 'SANOFI.NS', 'MERCK.NS', 'JBCHEPHARM.NS',
            'STRIDES.NS', 'CAPLIN.NS', 'LAURUSLABS.NS', 'SUVEN.NS',
            'WOCKPHARMA.NS', 'AUROPHARMA.NS', 'ZYDUSLIFE.NS',
            'GRANULES.NS', 'METROPOLIS.NS', 'THYROCARE.NS',
            'KIMS.NS', 'RAINBOW.NS', 'TEJASNET.NS',
            'HINDCOPPER.NS', 'MAZAGON.NS',
            'NYKAA.NS', 'CARTRADE.NS', 'EASEMYTRIP.NS', 'RVNL.NS',
            'RAILVIKAS.NS', 'IREDA.NS', 'POWERINDIA.NS',
            'ADANIGAS.NS', 'LINDEINDIA.NS', 'PRAXAIR.NS', 'INOXAIR.NS',
            'BASF.NS', 'AKZOINDIA.NS', 'BERGER.NS',
            'ASTER.NS', 'NARAYANANHL.NS', 'CIGNITI.NS', 'INDIGO.NS',
            'RADICO.NS', 'GLOBUSSPR.NS', 'RAYMOND.NS',
            'UJJIVAN.NS', 'SYMPHONY.NS', 'RAJESHEXPO.NS', 'ASTRAL.NS',
            'CERA.NS', 'VINATIORGA.NS', 'FSL.NS', 'CARBORUNIV.NS',
            'NRBBEARING.NS', 'BATA.NS', 'RELAXO.NS', 'LIBERTY.NS',
            'BLUESTARCO.NS', 'KEI.NS', 'POLYCAB.NS',
            'FINOLEX.NS', 'ABB.NS', 'SCHNEIDER.NS', 'HONAUT.NS', 'BHEL.NS',
            'BEML.NS', 'BEL.NS', 'HAL.NS', 'COCHINSHIP.NS', 'MAZAGON.NS',
        ]

        self.small_cap_stocks = [
            'CRISIL.NS', 'EQUITAS.NS', 'CDSL.NS', 'BSE.NS',
            'MCX.NS', 'NAZARA.NS', 'ONMOBILE.NS',
            'TV18BRDCST.NS', 'DISHTV.NS', 'SUNTV.NS', 'DHANUKA.NS', 'RALLIS.NS',
            'GHCL.NS', 'MANAPPURAM.NS', 'MUTHOOTFIN.NS',
            'SRTRANSFIN.NS', 'SHRIRAMFIN.NS', 'ESAFSFB.NS', 'SURYODAY.NS', 'FINPIPE.NS',
            'CAMS.NS', 'CARERATING.NS', 'ICRA.NS',
            'MOTILALOF.NS', 'ANGELONE.NS', 'IIFL.NS', 'GEOJITFSL.NS', 'VENKEYS.NS',
            'KRBL.NS', 'AMBER.NS', 'MINDA.NS',
            'SUNDARAM.NS', 'GLENMARK.NS', 'CADILAHC.NS', 'ALKEM.NS', 'AJANTPHARM.NS',
            'ABBOTINDIA.NS', 'PFIZER.NS', 'NOVARTIS.NS', 'SANOFI.NS',
            'MERCK.NS', 'JBCHEPHARM.NS', 'STRIDES.NS', 'CAPLIN.NS', 'LAURUSLABS.NS',
            'SUVEN.NS', 'WOCKPHARMA.NS', 'AUROPHARMA.NS', 'ZYDUSLIFE.NS',
            'GRANULES.NS', 'METROPOLIS.NS', 'THYROCARE.NS',
            'KIMS.NS', 'RAINBOW.NS', 'TEJASNET.NS',
            'HINDCOPPER.NS', 'MAZAGON.NS',
            'NYKAA.NS', 'CARTRADE.NS', 'EASEMYTRIP.NS', 'RVNL.NS',
            'RAILVIKAS.NS', 'IREDA.NS', 'SJVN.NS',
            'POWERINDIA.NS', 'ADANIGAS.NS', 'LINDEINDIA.NS', 'PRAXAIR.NS', 'INOXAIR.NS',
            'BASF.NS', 'AKZOINDIA.NS', 'BERGER.NS',
            'ASTER.NS', 'NARAYANANHL.NS', 'CIGNITI.NS',
            'RADICO.NS', 'GLOBUSSPR.NS', 'RAYMOND.NS',
            'UJJIVAN.NS', 'SYMPHONY.NS', 'RAJESHEXPO.NS', 'ASTRAL.NS',
            'CERA.NS', 'VINATIORGA.NS', 'FSL.NS', 'CARBORUNIV.NS',
            'NRBBEARING.NS',
        ]

    def get_all_stock_symbols(self):
        all_symbols = (
            self.large_cap_stocks +
            self.mid_cap_stocks +
            self.small_cap_stocks
        )
        return sorted(list(set(all_symbols)))

    def get_stock_by_category(self, category):
        if category == "Large Cap":
            return self.large_cap_stocks
        elif category == "Mid Cap":
            return self.mid_cap_stocks
        elif category == "Small Cap":
            return self.small_cap_stocks
        return []


stock_analyzer = StockAnalyzer()
all_stock_symbols = stock_analyzer.get_all_stock_symbols()
