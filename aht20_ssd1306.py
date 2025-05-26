#
# geminiのコードを、grokで最適化
#

import time
import datetime
from i2cpy import I2C
from typing import Tuple, Optional, Union, List, ByteString # Removed PIL and os imports
from ssd1306_driver import SSD1306, I2CError, i2c_write, i2c_read, DEFAULT_FONT_PATH, DEFAULT_FONT_SIZE


# 定数
AHT20_ADDRESS = 0x38
AHT20_TRIGGER_MEASUREMENT = bytes([0xAC, 0x33, 0x00])
# SSD1306 related constants are now in ssd1306_driver.py


class BMP280:
    """BMP280気圧・温度センサーの制御クラス"""
    ADDRESS = 0x77
    REGISTERS = {
        'DIG_T1': 0x88, 'CHIP_ID': 0xD0, 'RESET': 0xE0,
        'STATUS': 0xF3, 'CTRL_MEAS': 0xF4, 'CONFIG': 0xF5, 'PRESS_MSB': 0xF7
    }
    OVERSAMPLING = {'SKIP': 0x00, 'X1': 0x01, 'X2': 0x02, 'X4': 0x03, 'X8': 0x04, 'X16': 0x05}
    MODES = {'SLEEP': 0x00, 'FORCED': 0x01, 'NORMAL': 0x03}

    def __init__(self, i2c: I2C, address: int = ADDRESS):
        self.i2c = i2c
        self.address = address
        self._t_fine = 0
        self._calibration = {}
        self._load_calibration()
        self._configure()

    def _load_calibration(self) -> None:
        """キャリブレーションデータを読み込む"""
        data = i2c_read(self.i2c, self.address, self.REGISTERS['DIG_T1'], 24, "BMP280 calibration read")
        self._calibration.update({
            'dig_T1': (data[1] << 8) | data[0],
            'dig_T2': self._to_s16((data[3] << 8) | data[2]),
            'dig_T3': self._to_s16((data[5] << 8) | data[4]),
            'dig_P1': (data[7] << 8) | data[6],
            'dig_P2': self._to_s16((data[9] << 8) | data[8]),
            'dig_P3': self._to_s16((data[11] << 8) | data[10]),
            'dig_P4': self._to_s16((data[13] << 8) | data[12]),
            'dig_P5': self._to_s16((data[15] << 8) | data[14]),
            'dig_P6': self._to_s16((data[17] << 8) | data[16]),
            'dig_P7': self._to_s16((data[19] << 8) | data[18]),
            'dig_P8': self._to_s16((data[21] << 8) | data[20]),
            'dig_P9': self._to_s16((data[23] << 8) | data[22])
        })

    def _to_s16(self, value: int) -> int:
        """符号付き16ビット整数に変換"""
        return value - 65536 if value > 32767 else value

    def _configure(self, temp_os: int = OVERSAMPLING['X1'], press_os: int = OVERSAMPLING['X1'],
                   mode: int = MODES['NORMAL'], t_sb: int = 0x05, filter_coeff: int = 0x00) -> None:
        """センサーの測定設定を行う"""
        ctrl_meas = (temp_os << 5) | (press_os << 2) | mode
        config = (t_sb << 5) | (filter_coeff << 2)
        i2c_write(self.i2c, self.address, bytes([self.REGISTERS['CTRL_MEAS'], ctrl_meas]), "BMP280 ctrl_meas")
        i2c_write(self.i2c, self.address, bytes([self.REGISTERS['CONFIG'], config]), "BMP280 config")

    def _read_raw(self) -> Tuple[int, int]:
        """生の温度と気圧データを読み込む"""
        data = i2c_read(self.i2c, self.address, self.REGISTERS['PRESS_MSB'], 6, "BMP280 raw data")
        adc_p = (data[0] << 12) | (data[1] << 4) | (data[2] >> 4)
        adc_t = (data[3] << 12) | (data[4] << 4) | (data[5] >> 4)
        return adc_t, adc_p

    def _compensate_temperature(self, adc_t: int) -> float:
        """温度を補正して計算（℃）"""
        t1, t2, t3 = self._calibration['dig_T1'], self._calibration['dig_T2'], self._calibration['dig_T3']
        var1 = (adc_t / 16384.0 - t1 / 1024.0) * t2
        var2 = ((adc_t / 131072.0 - t1 / 8192.0) ** 2) * t3
        self._t_fine = var1 + var2
        return self._t_fine / 5120.0

    def _compensate_pressure(self, adc_p: int) -> float:
        """気圧を補正して計算（Pa）"""
        if self._t_fine == 0:
            self._compensate_temperature(self._read_raw()[0])
        p = self._calibration
        var1 = (self._t_fine / 2.0) - 64000.0
        var2 = var1 * var1 * p['dig_P6'] / 32768.0 + var1 * p['dig_P5'] * 2.0
        var2 = (var2 / 4.0) + (p['dig_P4'] * 65536.0)
        var3 = (p['dig_P3'] * var1 * var1 / 524288.0 + p['dig_P2'] * var1) / 524288.0
        var3 = (1.0 + var3 / 32768.0) * p['dig_P1']
        if var3 == 0:
            return 0.0
        pressure = 1048576.0 - adc_p
        pressure = (pressure - (var2 / 4096.0)) * 6250.0 / var3
        var4 = p['dig_P9'] * pressure * pressure / 2147483648.0
        var5 = pressure * p['dig_P8'] / 32768.0
        return pressure + (var4 + var5 + p['dig_P7']) / 16.0

    def read_temperature_pressure(self) -> Tuple[float, float]:
        """温度（℃）と気圧（Pa）を読み取る"""
        adc_t, adc_p = self._read_raw()
        temp = self._compensate_temperature(adc_t)
        pressure = self._compensate_pressure(adc_p)
        return temp, pressure

    def get_chip_id(self) -> int:
        """チップIDを読み取る（通常0x58）"""
        return i2c_read(self.i2c, self.address, self.REGISTERS['CHIP_ID'], 1, "BMP280 chip ID")[0]

# SSD1306 class and its helpers are now in ssd1306_driver.py

def read_aht20(i2c: I2C, address: int = AHT20_ADDRESS, trigger: bytes = AHT20_TRIGGER_MEASUREMENT) -> Tuple[float, float]:
    """AHT20から湿度と温度を読み取る"""
    i2c_write(i2c, address, trigger, "AHT20 trigger")
    time.sleep(0.08)
    data = i2c.readfrom(address, 6)
    humidity_raw = ((data[1] << 12) | (data[2] << 4) | (data[3] >> 4)) & 0xFFFFF
    humidity = round((humidity_raw * 100.0) / 0x100000,1)
    temp_raw = ((data[3] & 0x0F) << 16) | (data[4] << 8) | data[5]
    return (humidity_raw * 100.0 / 2**20), ((temp_raw * 200.0 / 2**20) - 50.0)

def main():
    """メイン処理"""
    try:
        i2c = I2C(driver="ch341")
        bmp280 = BMP280(i2c)
        display = SSD1306(i2c)

        chip_id = bmp280.get_chip_id()
        print(f"BMP280 Chip ID: 0x{chip_id:02X} {'(OK)' if chip_id == 0x58 else '(Unexpected)'}")

        if not display._init_display() or not display.clear():
            raise RuntimeError("Display setup failed")

        while True:
            now = datetime.datetime.now()
            formatted_datetime = now.strftime("%m/%d %H:%M:%S")
            # Use font_size parameter as before, font_path will use driver's default
            display.display_text(formatted_datetime, 0, 0, font_size=16)
            try:
                humidity, temp_aht = read_aht20(i2c)
                # These calls will use the default font size from the driver
                display.display_text(f"Temp: {temp_aht:.1f} C", 0, 20)
                display.display_text(f"Humi: {humidity:.1f} %", 0, 38)
                #print(f"AHT20: Temp = {temp_aht:.2f} °C, Humidity = {humidity:.2f} %")
            except I2CError as e:
                print(f"AHT20 read error: {e}")
    
            try:
                temp_bmp, pressure = bmp280.read_temperature_pressure()
                pressure_hpa = pressure / 100.0
                display.display_text(f"Pres: {pressure_hpa:.1f} hPa", 0, 50)
                #print(f"BMP280: Temp = {temp_bmp:.2f} °C, Pressure = {pressure_hpa:.2f} hPa")
            except I2CError as e:
                print(f"BMP280 read error: {e}")
    
            time.sleep(1)

    except Exception as e:
        print(f"Done: {e}")

if __name__ == "__main__":
    main()
