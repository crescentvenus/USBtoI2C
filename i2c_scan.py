from i2cpy import I2C

i2c = I2C(driver="ch341")
# I2Cバス上のデバイスをスキャン
devices = i2c.scan()
print("検出されたI2Cデバイスアドレス:", [hex(addr) for addr in devices])
