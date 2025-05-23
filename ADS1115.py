import time
import os
from typing import Tuple, ByteString
from i2cpy import I2C
from PIL import Image, ImageDraw, ImageFont

# 定数
ADS1115_ADDRESS = 0x48  # ADDRピンがGND
SSD1306_ADDRESS = 0x3C
DISPLAY_WIDTH = 128
DISPLAY_HEIGHT = 64
DEFAULT_FONT_PATH = "C:\\Windows\\Fonts\\msgothic.ttc"
DEFAULT_FONT_SIZE = 12

# SSD1306コマンド
SSD1306_COMMAND = 0x00
SSD1306_DATA = 0x40
SSD1306_INIT = [
    0xAE, 0xD5, 0x80, 0xA8, DISPLAY_HEIGHT - 1, 0xD3, 0x00, 0x40,
    0x8D, 0x14, 0xA1, 0xC8, 0xDA, 0x12, 0x81, 0xCF, 0xD9, 0xF1,
    0xDB, 0x40, 0xA4, 0xA6, 0xAF
]

# ADS1115レジスタ
ADS1115_REG_CONFIG = 0x01
ADS1115_REG_CONVERSION = 0x00

class I2CError(Exception):
    """I2C通信エラーのカスタム例外"""
    pass

def i2c_write(i2c: I2C, address: int, buffer: ByteString, operation: str = "write") -> None:
    """I2Cに書き込み"""
    try:
        i2c.writeto(address, buffer)
    except Exception as e:
        raise I2CError(f"{operation} failed at 0x{address:02x}: {e}")

def i2c_read(i2c: I2C, address: int, register: int, length: int, operation: str = "read") -> bytes:
    """I2Cから読み込み"""
    try:
        i2c.writeto(address, bytes([register & 0xFF]))
        return i2c.readfrom(address, length)
    except Exception as e:
        raise I2CError(f"{operation} failed at 0x{address:02x}, reg 0x{register:02x}: {e}")

class SSD1306:
    """SSD1306 OLEDディスプレイ制御"""
    def __init__(self, i2c: I2C, address: int = SSD1306_ADDRESS, width: int = DISPLAY_WIDTH, height: int = DISPLAY_HEIGHT):
        self.i2c = i2c
        self.address = address
        self.width = width
        self.height = height
        if not self._init_display():
            raise RuntimeError("SSD1306 initialization failed")

    def _init_display(self) -> bool:
        """ディスプレイを初期化"""
        try:
            for cmd in SSD1306_INIT:
                i2c_write(self.i2c, self.address, bytes([SSD1306_COMMAND, cmd]), "SSD1306 init")
            time.sleep(0.1)
            return True
        except I2CError as e:
            print(f"SSD1306 init error: {e}")
            return False

    def clear(self) -> bool:
        """ディスプレイをクリア"""
        try:
            commands = [
                0x20, 0x00,  # 水平アドレッシング
                0x21, 0, self.width - 1,  # カラム範囲
                0x22, 0, (self.height // 8) - 1  # ページ範囲
            ]
            for cmd in commands:
                i2c_write(self.i2c, self.address, bytes([SSD1306_COMMAND, cmd]), "SSD1306 clear")
            chunk_size = 16
            for _ in range(0, self.width * (self.height // 8), chunk_size):
                i2c_write(self.i2c, self.address, bytes([SSD1306_DATA]) + bytes([0x00] * chunk_size), "SSD1306 clear data")
            return True
        except I2CError as e:
            print(f"SSD1306 clear error: {e}")
            return False

    def _image_to_bytes(self, image: Image.Image) -> bytearray:
        """画像をSSD1306用バイト列に変換"""
        width, height = image.size
        pixels = image.load()
        data = bytearray()
        for page in range(height // 8):
            for x in range(width):
                byte = 0
                for bit in range(8):
                    y = page * 8 + bit
                    if y < height and pixels[x, y]:
                        byte |= 1 << bit
                data.append(byte)
        return data

    def display_text(self, text: str, x: int, y: int, font_path: str = DEFAULT_FONT_PATH, font_size: int = DEFAULT_FONT_SIZE) -> bool:
        """テキストを表示"""
        try:
            font = ImageFont.truetype(font_path, font_size) if os.path.exists(font_path) else ImageFont.load_default()
            bbox = font.getbbox(text)
            text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
            if text_width <= 0 or text_height <= 0:
                return True

            img = Image.new('1', (text_width, text_height), 0)
            draw = ImageDraw.Draw(img)
            draw.text((-bbox[0], -bbox[1]), text, font=font, fill=1)

            padded_height = ((text_height + 7) // 8) * 8
            padded_img = Image.new('1', (text_width, padded_height), 0)
            padded_img.paste(img, (0, 0))

            data = self._image_to_bytes(padded_img)
            col_start = max(0, x)
            col_end = min(self.width - 1, x + text_width - 1)
            page_start = max(0, y // 8)
            page_end = min((self.height // 8) - 1, (y + padded_height - 1) // 8)
            if col_start > col_end or page_start > page_end:
                return True

            commands = [
                0x20, 0x00, 0x21, col_start, col_end, 0x22, page_start, page_end
            ]
            for cmd in commands:
                i2c_write(self.i2c, self.address, bytes([SSD1306_COMMAND, cmd]), "SSD1306 text setup")

            display_width = col_end - col_start + 1
            display_pages = page_end - page_start + 1
            final_data = bytearray()
            for page in range(display_pages):
                src_page = page_start + page - (y // 8)
                if not (0 <= src_page < padded_height // 8):
                    final_data.extend([0x00] * display_width)
                    continue
                offset = src_page * text_width + max(0, col_start - x)
                end = offset + min(display_width, text_width - max(0, col_start - x))
                final_data.extend(data[offset:end])
                if len(final_data) % display_width != 0:
                    final_data.extend([0x00] * (display_width - (len(final_data) % display_width)))

            i2c_write(self.i2c, self.address, bytes([SSD1306_DATA]) + final_data, "SSD1306 text data")
            return True
        except (I2CError, IOError) as e:
            print(f"Text display error: '{text}' - {e}")
            return False

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
