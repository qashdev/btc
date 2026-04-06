"""
Download Bitcoin Historical Data from Binance
Free data from 2017 to present - Perfect for backtesting!
"""

from binance_historical_data import BinanceDataDumper
from datetime import date

def main():
    print("=" * 60)
    print("  Bitcoin Historical Data Downloader")
    print("=" * 60)

    # Setup the dumper
    dumper = BinanceDataDumper(
        path_dir_where_to_dump="./historical_data",  # Will create this folder
        asset_class="spot",      # Spot trading (not futures)
        data_type="klines",      # OHLCV candlestick data
        data_frequency="5m",     # 5-minute candles (matches your bot)
    )

    print("\n📊 Downloading BTCUSDT 5-minute data from Binance...")
    print("⏳ This will take 5-10 minutes depending on your internet speed...")
    print("\nDate range: 2020-01-01 to present")
    print("Why 2020? Good balance of data volume vs download time")
    print("(You can change dates later if needed)\n")

    # Download the data
    dumper.dump_data(
        tickers=["BTCUSDT"],
        date_start=date(2020, 1, 1),  # Start from 2020 (6+ years of data)
        date_end=None,                 # Until latest available
        is_to_update_existing=False,
    )

    print("\n" + "=" * 60)
    print("✅ Download Complete!")
    print("=" * 60)
    print("\n📁 Data saved in: ./historical_data/")
    print("📝 You'll see monthly CSV files like:")
    print("   - BTCUSDT-5m-2020-01.csv")
    print("   - BTCUSDT-5m-2020-02.csv")
    print("   - ... and so on")
    print("\n🚀 Next: Run the backtesting script!")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()