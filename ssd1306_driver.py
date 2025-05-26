import time
import os
from i2cpy import I2C
from PIL import Image, ImageDraw, ImageFont
from typing import ByteString

# 定数
SSD1306_ADDR = 0x3C
SSD1306_COMMAND = 0x00
SSD1306_DATA = 0x40
DISPLAY_WIDTH = 128
DISPLAY_HEIGHT = 64
# DEFAULT_FONT_PATH and DEFAULT_FONT_SIZE are likely application-specific
# and should ideally be passed to display_text or configured.
# For now, we'll keep them here but acknowledge this might need adjustment.

_font_cache = {}
DEFAULT_FONT_PATH = "C:\\Windows\\Fonts\\msgothic.ttc"
DEFAULT_FONT_SIZE = 16

# SSD1306初期化コマンド
SSD1306_INIT_COMMANDS = [
    0xAE, 0xD5, 0x80, 0xA8, DISPLAY_HEIGHT - 1, 0xD3, 0x00, 0x40,
    0x8D, 0x14, 0xA1, 0xC8, 0xDA, 0x12, 0x81, 0xCF, 0xD9, 0xF1,
    0xDB, 0x40, 0xA4, 0xA6, 0xAF
]

class I2CError(Exception):
    """I2C通信エラーを表すカスタム例外"""
    pass

def i2c_write(i2c: I2C, address: int, buffer: ByteString, operation: str = "write") -> None:
    """I2Cデバイスにデータを書き込む"""
    try:
        i2c.writeto(address, buffer)
    except Exception as e:
        raise I2CError(f"{operation} error at 0x{address:02x}: {e}")

def i2c_read(i2c: I2C, address: int, register: int, length: int, operation: str = "read") -> bytes:
    """I2Cデバイスからデータを読み込む"""
    try:
        i2c.writeto(address, bytes([register & 0xFF]))
        return i2c.readfrom(address, length)
    except Exception as e:
        raise I2CError(f"{operation} error at 0x{address:02x}, reg 0x{register:02x}: {e}")

class SSD1306:
    """SSD1306 OLEDディスプレイの制御クラス"""
    def __init__(self, i2c: I2C, address: int = SSD1306_ADDR, width: int = DISPLAY_WIDTH, height: int = DISPLAY_HEIGHT):
        self.i2c = i2c
        self.address = address
        self.width = width
        self.height = height
        self._init_display()

    def _init_display(self) -> bool:
        """ディスプレイを初期化"""
        try:
            for cmd in SSD1306_INIT_COMMANDS:
                i2c_write(self.i2c, self.address, bytes([SSD1306_COMMAND, cmd]), "SSD1306 command")
            time.sleep(0.1)
            return True
        except I2CError as e:
            print(f"Display initialization failed: {e}")
            return False

    def clear(self) -> bool:
        """ディスプレイをクリア"""
        try:
            commands = [
                0x20, 0x00,  # Horizontal addressing mode
                0x21, 0, self.width - 1,  # Column address
                0x22, 0, (self.height // 8) - 1  # Page address
            ]
            for cmd in commands:
                i2c_write(self.i2c, self.address, bytes([SSD1306_COMMAND, cmd]), "SSD1306 clear setup")
            
            # Clear display by writing zeros to each page
            # Each page is self.width bytes wide.
            # The number of pages is self.height // 8.
            # No need to iterate page by page with commands if we're just blasting zeros
            # after setting horizontal addressing mode.
            # The commands 0x21 and 0x22 set the column and page address range.
            # So we can send all data in one go if the I2C buffer allows, or page by page.
            # For safety and common I2C buffer limits, sending page by page is better.
            num_pages = self.height // 8
            for page in range(num_pages):
                # Set page address for each page (though 0x22 already set the range)
                # This is redundant if 0x22 correctly set the range for auto-increment.
                # However, some displays might need explicit page setting.
                # For SSD1306, after setting horizontal mode and full column/page range,
                # a continuous stream of data should fill the GDDRAM.
                # Let's stick to the original intent of iterating "pages" but send full width.
                # The original loop was: for _ in range(0, self.width * (self.height // 8), chunk_size):
                # This implies sending self.width * (self.height // 8) bytes in total.
                # The new approach is to send self.width bytes for each of the (self.height // 8) pages.
                # This is effectively what setting column and page start/end addresses does.
                # We can iterate page by page and send self.width zeros.
                # This requires setting the page start address for each iteration.
                # The commands list already set column start/end and page start/end.
                # So, we can just send all data.
                pass # Commands already set the address window.

            # Send all data for all pages
            # Total bytes = self.width * (self.height // 8)
            # This might be too large for a single i2c_write.
            # Let's send data page by page.
            for p in range(self.height // 8):
                # Optionally, set current page address if display does not auto-increment pages well
                # i2c_write(self.i2c, self.address, bytes([SSD1306_COMMAND, 0xB0 | p]), "SSD1306 set page")
                # i2c_write(self.i2c, self.address, bytes([SSD1306_COMMAND, 0x00]), "SSD1306 set col low")
                # i2c_write(self.i2c, self.address, bytes([SSD1306_COMMAND, 0x10]), "SSD1306 set col high")
                i2c_write(self.i2c, self.address, bytes([SSD1306_DATA]) + bytes([0x00] * self.width), "SSD1306 clear page data")
            return True
        except I2CError as e:
            print(f"Display clear failed: {e}")
            return False

    def _image_to_bytes(self, image: Image.Image) -> bytearray:
        """Pillow画像をSSD1306用バイト列に変換"""
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

    def display_text(self, text: str, x: int, y: int, color: int = 1,
                     font_path: str = DEFAULT_FONT_PATH, font_size: int = DEFAULT_FONT_SIZE) -> bool:
        """テキストを表示"""
        global _font_cache
        cache_key = (font_path, font_size)

        if cache_key in _font_cache:
            font = _font_cache[cache_key]
        else:
            try:
                if font_path and os.path.exists(font_path):
                    font = ImageFont.truetype(font_path, font_size)
                else:
                    if font_path and font_path != DEFAULT_FONT_PATH: # Print warning only if a specific non-default path was given and not found
                        print(f"Warning: Font path '{font_path}' not found. Using default font.")
                    font = ImageFont.load_default()
            except IOError: # Handle cases where truetype loading fails for an existing path
                print(f"Warning: Could not load font at '{font_path}'. Using default font.")
                font = ImageFont.load_default()
            
            _font_cache[cache_key] = font
            
        try:
            bbox = font.getbbox(text)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            if text_width <= 0 or text_height <= 0:
                return True

            img = Image.new('1', (text_width, text_height), 0)
            draw = ImageDraw.Draw(img)
            draw.text((-bbox[0], -bbox[1]), text, font=font, fill=color)

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
                0x20, 0x00,
                0x21, col_start, col_end,
                0x22, page_start, page_end
            ]
            for cmd in commands:
                i2c_write(self.i2c, self.address, bytes([SSD1306_COMMAND, cmd]), "SSD1306 text setup")

            display_width_bytes = col_end - col_start + 1
            display_pages = page_end - page_start + 1
            final_data = bytearray()

            for page_idx in range(display_pages):
                # Calculate the source page in the padded_img's coordinate system
                # y is the top-left corner of the text box on the display
                # page_start is the first page row on the display we're writing to
                # current_display_page_abs = page_start + page_idx (absolute page index on display)
                # src_page_in_padded_img = current_display_page_abs - (y // 8) (page index relative to padded_img's top)
                
                src_page_in_padded_img = (page_start + page_idx) - (y // 8)

                if not (0 <= src_page_in_padded_img < (padded_height // 8)):
                    # This part of the display is outside the text's vertical extent
                    final_data.extend([0x00] * display_width_bytes)
                    continue

                # Calculate source x start within the text_width
                # col_start is the starting column on the display
                # x is the starting column of the text box on the display
                src_x_start_in_text = max(0, col_start - x)
                
                # Calculate how many bytes to copy from this row of the text image
                bytes_to_copy_this_row = min(display_width_bytes, text_width - src_x_start_in_text)
                
                if bytes_to_copy_this_row <= 0:
                    final_data.extend([0x00] * display_width_bytes)
                    continue

                offset_in_data = src_page_in_padded_img * text_width + src_x_start_in_text
                final_data.extend(data[offset_in_data : offset_in_data + bytes_to_copy_this_row])

                # Pad if the text is narrower than the display window we're writing to
                if bytes_to_copy_this_row < display_width_bytes:
                    final_data.extend([0x00] * (display_width_bytes - bytes_to_copy_this_row))

            i2c_write(self.i2c, self.address, bytes([SSD1306_DATA]) + final_data, "SSD1306 text data")
            return True
        except (I2CError, IOError, ValueError) as e: # Added ValueError for font issues
            print(f"Text display failed for '{text}': {e}")
            return False
