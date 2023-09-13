from sqlalchemy import create_engine, Table, Column, Integer, String, Float, Date
from sqlalchemy.ext.declarative import declarative_base
from tools.logger import Logger

logger = Logger(__name__).get_logger()
Base = declarative_base()

class StockBasicInfo(Base):
    __tablename__ = 'StockBasicInfo'
    
    id = Column(Integer, primary_key=True)  # ORM에서는 대부분의 경우 기본 키가 필요합니다.
    acml_tr_pbmn = Column(String(255))
    acml_vol = Column(Integer)
    askp = Column(Integer)
    bidp = Column(Integer)
    cpfn = Column(Integer)
    eps = Column(Float)
    hts_avls = Column(Integer)
    hts_kor_isnm = Column(String(255))
    itewhol_loan_rmnd_ratem_name = Column(Float)  # 원래의 이름에 공백이 있어서 이름을 수정했습니다.
    lstn_stcn = Column(Integer)
    pbr = Column(Float)
    per = Column(Float)
    prdy_ctrt = Column(Float)
    prdy_vol = Column(Integer)
    prdy_vrss = Column(Integer)
    prdy_vrss_sign = Column(Integer)
    prdy_vrss_vol = Column(Integer)
    stck_fcam = Column(Integer)
    stck_hgpr = Column(Integer)
    stck_llam = Column(Integer)
    stck_lwpr = Column(Integer)
    stck_mxpr = Column(Integer)
    stck_oprc = Column(Integer)
    stck_prdy_clpr = Column(Integer)
    stck_prdy_hgpr = Column(Integer)
    stck_prdy_lwpr = Column(Integer)
    stck_prdy_oprc = Column(Integer)
    stck_prpr = Column(Integer)
    stck_shrn_iscd = Column(String(255))
    vol_tnrt = Column(Float)

class StockDailyData(Base):
    __tablename__ = 'StockDailyData'

    id = Column(Integer, primary_key=True)  # 일반적으로 기본 키가 필요합니다.
    stock_code = Column(String(255))
    acml_tr_pbmn = Column(String(255))
    acml_vol = Column(String(255))
    flng_cls_code = Column(String(255))
    mod_yn = Column(String(255))
    prdy_vrss = Column(String(255))
    prdy_vrss_sign = Column(String(255))
    prtt_rate = Column(String(255))
    revl_issu_reas = Column(String(255))
    stck_bsop_date = Column(Date)
    stck_clpr = Column(String(255))
    stck_hgpr = Column(String(255))
    stck_lwpr = Column(String(255))
    stck_oprc = Column(String(255))
