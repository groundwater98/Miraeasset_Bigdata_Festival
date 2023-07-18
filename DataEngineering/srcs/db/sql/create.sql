CREATE TABLE IF NOT EXISTS stock_data (
    id SERIAL PRIMARY KEY,
    date TIMESTAMP WITH TIME ZONE,
    open NUMERIC,
    high NUMERIC,
    low NUMERIC,
    close NUMERIC,
    adj_close NUMERIC,
    volume BIGINT,
    dividends NUMERIC,
    stock_splits NUMERIC
);

CREATE TABLE IF NOT EXISTS stock_metadata (
    symbol VARCHAR(10) PRIMARY KEY,
    currency VARCHAR(10),
    exchange_name VARCHAR(50),
    instrument_type VARCHAR(20),
    first_trade_date TIMESTAMP,
    regular_market_time BIGINT,
    gmtoffset INTEGER,
    timezone VARCHAR(50),
    exchange_timezone_name VARCHAR(50),
    regular_market_price NUMERIC,
    chart_previous_close NUMERIC,
    price_hint INTEGER,
    data_granularity VARCHAR(10),
    valid_ranges VARCHAR[],
    current_trading_period JSONB
);

CREATE TABLE IF NOT EXISTS stock_company (
    symbol VARCHAR(10) PRIMARY KEY,
    company_name VARCHAR(100),
    long_name VARCHAR(100),
    short_name VARCHAR(100),
    long_business_summary VARCHAR(1000),
    short_business_summary VARCHAR(1000),
    industry VARCHAR(100),
    sector VARCHAR(100),
    full_time_employees INTEGER,
    company_website VARCHAR(100),
    company_logo VARCHAR(100),
    address1 VARCHAR(100),
    city VARCHAR(100),
    state VARCHAR(100),
    zip VARCHAR(100),
    country VARCHAR(100),
    phone VARCHAR(100),
    fax VARCHAR(100),
    email VARCHAR(100),
    market VARCHAR(100),
    financial_currency VARCHAR(100),
    last_updated TIMESTAMP
);
