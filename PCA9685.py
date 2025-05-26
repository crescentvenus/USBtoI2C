#
# Gemini 2.5 Proが提案したコード
#
import time
from i2cpy import I2C

class PCA9685:
    """
    PCA9685 PWM Servo/LED Controller Driver.
    Assumes i2cpy library provides an I2C object with a `writeto(address, buffer)` method
    and potentially a `readfrom(address, num_bytes)` method.
    """

    # Registers/etc.
    __MODE1              = 0x00
    __MODE2              = 0x01
    __PRESCALE           = 0xFE
    __LED0_ON_L          = 0x06
    __LED0_ON_H          = 0x07
    __LED0_OFF_L         = 0x08
    __LED0_OFF_H         = 0x09
    __ALL_LED_ON_L       = 0xFA
    __ALL_LED_ON_H       = 0xFB
    __ALL_LED_OFF_L      = 0xFC
    __ALL_LED_OFF_H      = 0xFD

    # MODE1 bits
    __RESTART            = 0x80
    __SLEEP              = 0x10
    __ALLCALL            = 0x01
    __AI                 = 0x20 # Auto-Increment, useful for multi-byte writes

    def __init__(self, i2c_bus, address=0x40, osc_freq=25000000):
        """
        Initialize the PCA9685.
        :param i2c_bus: The i2cpy I2C bus object.
        :param address: The I2C address of the PCA9685 (default 0x40).
        :param osc_freq: The frequency of the internal oscillator in Hz (default 25MHz).
        """
        self.i2c = i2c_bus
        self.address = address
        self._osc_freq = osc_freq
        self._initialize()

    def _write_byte_data(self, register, value):
        """Writes an 8-bit value to the specified register."""
        buffer = bytes([register & 0xFF, value & 0xFF])
        try:
            self.i2c.writeto(self.address, buffer)
        except Exception as e:
            print(f"Error writing to PCA9685 reg {register:#04x} with value {value:#04x}: {e}")
            raise

    def _read_byte_data(self, register):
        """
        Reads an 8-bit value from the specified register.
        This method's success depends on i2cpy's capabilities.
        """
        buffer_reg = bytes([register & 0xFF])
        try:
            # Step 1: Write the register address to set the pointer
            self.i2c.writeto(self.address, buffer_reg)
            # Step 2: Read 1 byte from the device
            # This assumes i2cpy has a method like 'readfrom(address, num_bytes)'
            read_value = self.i2c.readfrom(self.address, 1)
            return read_value[0]
        except AttributeError:
            # i2c.readfrom is not available or has a different signature
            print(f"Warning: i2c.readfrom not available or failed for i2cpy. Cannot read PCA9685 register {register:#04x}.")
            # Fallback: return a default value or raise specific error
            # For MODE1, a typical value after init might include AI and ALLCALL
            if register == self.__MODE1:
                # Return a common state if read fails, assuming AI and ALLCALL are set, and device is awake.
                return self.__AI | self.__ALLCALL
            return 0 # Or raise an error indicating read failure
        except Exception as e:
            print(f"Error reading from PCA9685 reg {register:#04x}: {e}")
            raise # Or return a sensible default

    def _initialize(self):
        """Initializes the PCA9685 chip."""
        # Set all PWM channels to off
        self.set_all_pwm(0, 0)
        # Configure MODE2 for totem pole output structure (common for LEDs/Servos)
        self._write_byte_data(self.__MODE2, 0x04) # OUTDRV=1

        try:
            mode1 = self._read_byte_data(self.__MODE1)
            mode1 &= ~self.__SLEEP  # Wake up (clear sleep bit)
            mode1 |= self.__AI      # Enable Auto-Increment
            mode1 |= self.__ALLCALL # Enable AllCall address
            self._write_byte_data(self.__MODE1, mode1)
        except Exception: # If read fails or subsequent write fails
            print("Warning: Failed to read/write MODE1 during init. Setting default MODE1 (awake, AI, AllCall).")
            self._write_byte_data(self.__MODE1, self.__AI | self.__ALLCALL)

        time.sleep(0.005) # Wait for oscillator to stabilize

    def set_pwm_freq(self, freq_hz):
        """
        Sets the PWM frequency.
        :param freq_hz: The desired frequency in Hertz (typically 24Hz to 1526Hz).
        """
        if not (24 <= freq_hz <= 1526):
            print(f"Warning: freq_hz {freq_hz} is outside the typical range (24-1526 Hz).")

        prescaleval = float(self._osc_freq)
        prescaleval /= 4096.0  # 12-bit resolution
        prescaleval /= float(freq_hz)
        prescaleval -= 1.0
        prescale = int(round(prescaleval))

        if prescale < 3: prescale = 3       # Minimum prescale value
        if prescale > 255: prescale = 255   # Maximum prescale value

        try:
            old_mode = self._read_byte_data(self.__MODE1)
            # Go to sleep before setting the prescaler
            self._write_byte_data(self.__MODE1, (old_mode & ~self.__RESTART) | self.__SLEEP)
            self._write_byte_data(self.__PRESCALE, prescale)
            # Restore old mode (this will also wake it up if it was originally awake)
            self._write_byte_data(self.__MODE1, old_mode & ~self.__SLEEP) # Ensure SLEEP is off
            time.sleep(0.005) # Wait for oscillator
            # Restart PWM channels with the new frequency
            self._write_byte_data(self.__MODE1, old_mode | self.__RESTART)

        except Exception as e_freq_set:
            print(f"Warning: Could not read/write MODE1 during freq set ({e_freq_set}). Using fallback.")
            # Fallback if read failed or write failed
            self._write_byte_data(self.__MODE1, self.__AI | self.__ALLCALL | self.__SLEEP)
            self._write_byte_data(self.__PRESCALE, prescale)
            self._write_byte_data(self.__MODE1, self.__AI | self.__ALLCALL | self.__RESTART) # SLEEP cleared, RESTART set

        time.sleep(0.005) # Wait for oscillator to stabilize after frequency change.

    def set_pwm(self, channel, on_ticks, off_ticks):
        """
        Sets a single PWM channel.
        :param channel: Channel number (0-15).
        :param on_ticks: The tick value (0-4095) when the signal should go high.
        :param off_ticks: The tick value (0-4095) when the signal should go low.
                          Special values 4096 can be used for always on/off (see set_duty_cycle).
        """
        if not (0 <= channel <= 15):
            raise ValueError("Channel must be between 0 and 15.")
        if not (0 <= on_ticks <= 4096): # Allow 4096 for special always on/off cases
            raise ValueError("ON ticks must be between 0 and 4096.")
        if not (0 <= off_ticks <= 4096): # Allow 4096 for special always on/off cases
            raise ValueError("OFF ticks must be between 0 and 4096.")

        # The PCA9685 has 4 registers per channel: LEDn_ON_L, LEDn_ON_H, LEDn_OFF_L, LEDn_OFF_H
        # The first byte in the buffer is the starting register address.
        # Auto-Increment (AI bit in MODE1) should be enabled for this to work efficiently.
        reg_base = self.__LED0_ON_L + 4 * channel
        buffer = bytes([
            reg_base,
            on_ticks & 0xFF,        # ON_L
            (on_ticks >> 8) & 0xFF, # ON_H
            off_ticks & 0xFF,       # OFF_L
            (off_ticks >> 8) & 0xFF # OFF_H
        ])
        try:
            self.i2c.writeto(self.address, buffer)
        except Exception as e:
            print(f"Error writing PWM to PCA9685 channel {channel}: {e}")
            raise

    def set_all_pwm(self, on_ticks, off_ticks):
        """
        Sets all PWM channels to the same on/off tick values.
        Uses ALL_LED registers for efficiency.
        """
        if not (0 <= on_ticks <= 4096):
            raise ValueError("ON ticks must be between 0 and 4096.")
        if not (0 <= off_ticks <= 4096):
            raise ValueError("OFF ticks must be between 0 and 4096.")

        buffer = bytes([
            self.__ALL_LED_ON_L,
            on_ticks & 0xFF,
            (on_ticks >> 8) & 0xFF,
            off_ticks & 0xFF,
            (off_ticks >> 8) & 0xFF
        ])
        try:
            self.i2c.writeto(self.address, buffer)
        except Exception as e:
            print(f"Error writing to ALL_LED registers on PCA9685: {e}")
            raise

    def set_duty_cycle(self, channel, duty_cycle_percent):
        """
        Sets the PWM duty cycle for a specific channel.
        A duty cycle of 0% means always off, 100% means always on.
        :param channel: Channel number (0-15).
        :param duty_cycle_percent: Desired duty cycle (0.0 to 100.0).
        """
        if not (0.0 <= duty_cycle_percent <= 100.0):
            raise ValueError("Duty cycle must be between 0.0 and 100.0.")

        if duty_cycle_percent <= 0.0:
            # To force output fully OFF: LEDn_ON_H[4]=0, LEDn_OFF_H[4]=1
            # This means on_ticks=0, off_ticks=0x1000 (4096)
            self.set_pwm(channel, 0, 4096)
        elif duty_cycle_percent >= 100.0:
            # To force output fully ON: LEDn_ON_H[4]=1, LEDn_OFF_H[4]=0
            # This means on_ticks=0x1000 (4096), off_ticks=0
            self.set_pwm(channel, 4096, 0)
        else:
            # Standard PWM: pulse starts at tick 0 (on_ticks=0).
            # Pulse goes low at off_ticks.
            # off_ticks determines the pulse width.
            off_val = int(round(duty_cycle_percent * 40.95)) # Scale 0-100 to 0-4095
            if off_val > 4095: off_val = 4095
            if off_val < 0: off_val = 0
            self.set_pwm(channel, 0, off_val)

    def set_servo_pulse(self, channel, pulse_us, freq_hz=50):
        """
        Sets the PWM pulse width for a servo motor.
        :param channel: Channel number (0-15).
        :param pulse_us: Desired pulse width in microseconds (e.g., 1000 to 2000 for standard servos).
        :param freq_hz: The PWM frequency the servo expects (usually 50Hz).
        """
        # Calculate ticks from microseconds
        period_us = 1000000.0 / freq_hz
        ticks_per_us = 4096.0 / period_us
        off_ticks = int(round(pulse_us * ticks_per_us))

        if not (0 <= off_ticks <= 4095):
            # Clamp to valid range, or raise error, or print warning.
            # Here, we'll clamp and warn, but this might indicate an issue with pulse_us or freq_hz.
            print(f"Warning: Calculated off_ticks ({off_ticks}) for pulse {pulse_us}us at {freq_hz}Hz is out of 0-4095 range. Clamping.")
            if off_ticks < 0: off_ticks = 0
            if off_ticks > 4095: off_ticks = 4095
        
        self.set_pwm(channel, 0, off_ticks)

# --- Example Usage ---
if __name__ == "__main__":
    try:
        # 1. Initialize i2cpy (as you did for the SSD1306)
        # This is specific to your i2cpy setup.
        # from i2cpy import I2C # Make sure this matches your library import
        # i2c_bus = I2C(driver="ch341") # Example from your previous code

        i2c_bus = I2C(driver="ch341") # Replace with your actual i2c_bus object
        # --- End of mock setup ---

        # 2. Create an instance of the PCA9685 class
        # Default I2C address is 0x40. Change if your device uses a different address.
        pca = PCA9685(i2c_bus, address=0x40)

        # 3. Set the PWM frequency (e.g., 50Hz for servos, higher for LEDs)
        print("Setting PWM frequency to 50Hz for servos...")
        pca.set_pwm_freq(50) # For servos

        # --- Control a Servo Example (Channel 0) ---
        # Standard servos typically respond to pulses from ~500us to ~2500us,
        # with ~1500us being the center position.
        # These values can vary by servo model.
        servo_channel = 15 # <----------------------------------- adjust port #
        min_pulse_us = 600  # Adjust for your servo's minimum
        max_pulse_us = 2400 # Adjust for your servo's maximum
        center_pulse_us = (min_pulse_us + max_pulse_us) / 2

        print(f"Moving servo on channel {servo_channel} to min position...")
        pca.set_servo_pulse(servo_channel, min_pulse_us, freq_hz=50)
        time.sleep(1)

        print(f"Moving servo on channel {servo_channel} to center position...")
        pca.set_servo_pulse(servo_channel, center_pulse_us, freq_hz=50)
        time.sleep(10)

        print(f"Moving servo on channel {servo_channel} to max position...")
        pca.set_servo_pulse(servo_channel, max_pulse_us, freq_hz=50)
        time.sleep(1)
        

        print("Turning off all PWM channels...")
        pca.set_all_pwm(0, 0) # Turn all outputs off (or use (0, 4096) for full off state)

        print("PCA9685 example finished.")

    except ImportError:
        print("ImportError: Make sure i2cpy is installed and accessible.")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
