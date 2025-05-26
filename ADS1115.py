import time
from typing import Tuple, ByteString
from i2cpy import I2C
# Removed os and PIL imports as they are no longer used directly in this file.
from ssd1306_driver import SSD1306, I2CError, i2c_write, i2c_read, DEFAULT_FONT_PATH, DEFAULT_FONT_SIZE

# 定数
ADS1115_ADDRESS = 0x48  # ADDRピンがGND
# SSD1306 related constants (SSD1306_ADDRESS, DISPLAY_WIDTH, DISPLAY_HEIGHT,
# DEFAULT_FONT_PATH, DEFAULT_FONT_SIZE, SSD1306_COMMAND, SSD1306_DATA, SSD1306_INIT)
# are now managed by ssd1306_driver.py and imported if needed (e.g. DEFAULT_FONT values for SSD1306 class).

# ADS1115レジスタ
ADS1115_REG_CONFIG = 0x01
ADS1115_REG_CONVERSION = 0x00

# I2CError, i2c_write, i2c_read, and SSD1306 class are now imported from ssd1306_driver

class ADS1115:
    """ADS1115 16ビットADC制御"""
    ADDRESS = 0x48
    VREF = 4.096  # ゲイン1（±4.096V）
    CONFIG_MUX = {'A0': 0x4000, 'A1': 0x5000, 'A2': 0x6000, 'A3': 0x7000}  # シングルエンド
    CONFIG_BASE = 0x8000 | 0x0200 | 0x0080 | 0x0080 | 0x0003  # 開始, ゲイン1, シングルショット, 128SPS, コンパレータ無効

    def __init__(self, i2c: I2C, address: int = ADDRESS):
        self.i2c = i2c
        self.address = address

    def _read_channel(self, channel: str) -> float:
        """指定チャンネルの電圧を読み取る（V）"""
        config = self.CONFIG_BASE | self.CONFIG_MUX[channel]
        i2c_write(self.i2c, self.address, bytes([ADS1115_REG_CONFIG, (config >> 8) & 0xFF, config & 0xFF]), f"ADS1115 config {channel}")
        time.sleep(0.01)  # 128SPSで約8ms
        data = i2c_read(self.i2c, self.address, ADS1115_REG_CONVERSION, 2, f"ADS1115 read {channel}")
        raw = (data[0] << 8) | data[1]
        if raw & 0x8000:
            raw -= 0x10000
        return max(0.0, min((raw * self.VREF) / 32768.0, self.VREF))

    def read_voltages(self) -> Tuple[float, float, float, float]:
        """A0〜A3の電圧を読み取る"""
        return tuple(self._read_channel(ch) for ch in ['A0', 'A1', 'A2', 'A3'])

def main():
    """メイン処理"""
    try:
        i2c = I2C(driver="ch341")
        ads = ADS1115(i2c)
        display = SSD1306(i2c)
        print(f"ADS1115 initialized at 0x{ads.address:02X}")

        for i in range(50):
            print(f"\n--- Measurement {i+1} ---")
            display.clear()
            try:
                v0, v1, v2, v3 = ads.read_voltages()
                display.display_text(f"A0: {v0:.3f}V", 0, 0)
                display.display_text(f"A1: {v1:.3f}V", 0, 16)
                display.display_text(f"A2: {v2:.3f}V", 0, 32)
                display.display_text(f"A3: {v3:.3f}V", 0, 48)
                #print(f"ADS1115: A0={v0:.3f}V, A1={v1:.3f}V, A2={v2:.3f}V, A3={v3:.3f}V")
            except I2CError as e:
                print(f"ADS1115 error: {e}")
            time.sleep(2)

    except Exception as e:
        print(f"Main error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
