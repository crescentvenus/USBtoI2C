import time
import logging
import threading
import cv2
import numpy as np
from i2cpy import I2C
from ipywidgets import IntSlider, Button, HBox, VBox, Image, Label, Dropdown
from IPython.display import display
from ultralytics import YOLO # YOLOv8をインポート
import collections
import os # ファイル操作のためにインポート

# ロガーの設定
# 開発中はINFOまたはDEBUGに設定し、詳細なログを確認してください。
logging.basicConfig(
    level=logging.INFO,
    filename='object_tracking_debug.log',
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ServoError(Exception):
    """サーボ関連のカスタム例外"""
    pass

class Config:
    """サーボとUIの設定"""
    ANGLE_CENTER = 90
    PAN_ANGLE_RANGE = 60
    TILT_ANGLE_RANGE = 30
    MIN_PULSE_US = 500
    MAX_PULSE_US = 2500
    PWM_FREQ_HZ = 50
    MOVE_STEPS = 20
    MOVE_DURATION = 0.5
    CHANNELS = [15, 14]  # ch15: パン, ch14: チルト
    LOG_LEVEL = logging.INFO # ログレベル
    RETRIES = 3
    RETRY_DELAY = 0.2
    CAMERA_ID = 0
    FRAME_WIDTH = 1920
    FRAME_HEIGHT = 1080
    TRACKING_FPS = 15 # 追尾のフレームレート
    MAX_ANGLE_CHANGE = 10 # 1回の更新で動く最大角度
    CAMERA_FOV_X = 65  # カメラの水平画角（要実測）
    CAMERA_FOV_Y = 50  # カメラの垂直画角（要実測）
    DEBUG_FIX_TILT_ZERO = False # チルトをゼロに固定してデバッグするかどうか

    # PID制御のゲイン (システムの特性に合わせて調整が非常に重要です)
    # KP: 比例ゲイン (現在の誤差に比例して修正)
    # KI: 積分ゲイン (過去の誤差の蓄積を解消し、定常偏差をなくす)
    # KD: 微分ゲイン (誤差の変化率に反応し、オーバーシュートや振動を抑制)
    KP_PAN = 0.05
    KI_PAN = 0.001
    KD_PAN = 0.01
    KP_TILT = 0.05
    KI_TILT = 0.001
    KD_TILT = 0.01

    DEADBAND_PX_BASE = 5 # デッドバンドの基本値 (ピクセル)。この範囲内の誤差は無視される。
    MIN_ANGLE_DIFF = 0.1 # サーボが動くと判断する最小角度差

    # 検出対象カテゴリとYOLOモデル
    # YOLOv8 nano (yolov8n.pt) モデルがCOCOデータセットで学習したクラスID
    # これらのIDはモデルによって異なる場合があります。
    TARGET_OBJECTS = {
        "人": 0,
        "車": 2,
        "鳥": 14,
        "飛行機": 4
    }
    # 初期選択カテゴリのクラスID
    DEFAULT_TRACK_CLASS_ID = 0 # 初期値は「人」のクラスID
    # YOLOモデルのパス (初回実行時に自動ダウンロードされます)
    # yolov8n.pt (nano), yolov8s.pt (small), yolov8m.pt (medium) などがあります。
    # nは軽量ですが精度は低め、mはバランス型です。
    YOLO_MODEL_PATH = 'yolov8n.pt' 
    CONF_THRESHOLD = 0.5 # 検出信頼度閾値 (この値以上の信頼度を持つ検出のみを考慮)

    # 近傍検索の半径
    SEARCH_RADIUS_PX = 100 # ピクセル単位での近傍検索半径

    # 動画録画設定
    BUFFER_DURATION_SEC = 5 # 録画開始時に遡ってバッファする時間 (秒)
    BUFFER_START_DELAY_SEC = 3 # 追尾開始から録画開始までの時間 (X秒)
    DETECTION_FAIL_COUNT_THRESHOLD = 30 # Y回: 連続で検知できなければ録画停止
    MAX_RECORDING_DURATION_SEC = 60 # Z秒: 最大録画時間
    VIDEO_FPS = 30 # 録画FPS (カメラのFPSと一致させるか、それ以下に設定)
    VIDEO_CODEC = cv2.VideoWriter_fourcc(*'XVID') # コーデック (例: 'XVID' for .avi)
    VIDEO_FILENAME_PREFIX = 'tracking_video_' # 録画ファイル名のプレフィックス
    VIDEO_SAVE_DIR = 'recorded_videos' # 動画保存ディレクトリ

class PIDController:
    """PIDコントローラクラス"""
    def __init__(self, kp, ki, kd, output_limits=(-Config.MAX_ANGLE_CHANGE, Config.MAX_ANGLE_CHANGE), dt=1.0/Config.TRACKING_FPS):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limits = output_limits # 出力値の制限
        self.dt = dt # 制御周期のデフォルト値

        self._integral = 0 # 積分項の累積値
        self._last_error = 0 # 前回の誤差
        self._last_time = time.time() # 前回の更新時刻

        logger.debug(f"PIDController初期化: Kp={kp}, Ki={ki}, Kd={kd}, Output_Limits={output_limits}, dt={dt}")

    def update(self, error):
        """
        誤差に基づいてPID制御の出力を計算します。
        Args:
            error (float): 現在の誤差。
        Returns:
            float: 計算された制御出力。
        """
        current_time = time.time()
        # 実際に経過した時間に基づいてdtを調整し、よりロバストに
        actual_dt = current_time - self._last_time
        if actual_dt <= 0: # ゼロ除算回避、または非常に短い時間の場合
            actual_dt = self.dt # デフォルト値を使用
        
        # P項 (比例項): 現在の誤差に比例
        p_term = self.kp * error

        # I項 (積分項): 過去の誤差を累積し、定常偏差を解消
        self._integral += error * actual_dt
        # I項の飽和を防ぐ（ワインドアップ防止）
        # 積分項が大きくなりすぎると、目標値に到達した後もオーバーシュートが続くため、出力を制限
        if self.ki != 0: # kiがゼロでない場合のみ飽和チェック
            if self._integral > self.output_limits[1] / self.ki:
                self._integral = self.output_limits[1] / self.ki
            elif self._integral < self.output_limits[0] / self.ki:
                self._integral = self.output_limits[0] / self.ki
        i_term = self.ki * self._integral

        # D項 (微分項): 誤差の変化率に比例し、オーバーシュートを抑制
        derivative = (error - self._last_error) / actual_dt
        d_term = self.kd * derivative

        # PID出力の合計
        output = p_term + i_term + d_term

        # 出力制限 (サーボの最大角度変化量を超えないように)
        output = max(self.output_limits[0], min(self.output_limits[1], output))

        self._last_error = error # 次の計算のために現在の誤差を保存
        self._last_time = current_time # 次の計算のために現在の時刻を保存
        logger.debug(f"PID更新: Error={error:.2f}, P={p_term:.2f}, I={i_term:.2f}, D={d_term:.2f}, Output={output:.2f}, Actual_dt={actual_dt:.4f}")
        return output

    def reset(self):
        """PIDコントローラの内部状態をリセットします。"""
        self._integral = 0
        self._last_error = 0
        self._last_time = time.time()
        logger.debug("PIDControllerリセット")

class PCA9685:
    """PCA9685 PWMコントローラを制御するクラス"""
    _REG = {
        'MODE1': 0x00, 'MODE2': 0x01, 'PRESCALE': 0xFE,
        'LED0_ON_L': 0x06, 'ALL_LED_ON_L': 0xFA
    }
    _MODE1_FLAGS = {'RESTART': 0x80, 'SLEEP': 0x10, 'AI': 0x20, 'ALLCALL': 0x01}

    def __init__(self, i2c_bus, address=0x40, osc_freq=25000000):
        self.i2c = i2c_bus
        self.address = address
        self.osc_freq = osc_freq # 内部オシレータ周波数
        self._initialize()

    def _initialize(self):
        """PCA9685チップの初期化処理"""
        for attempt in range(Config.RETRIES):
            try:
                # 全てのPWM出力をオフに設定
                self._write_reg(self._REG['ALL_LED_ON_L'], bytes([0, 0, 0, 0]))
                # MODE2レジスタ設定 ( totem-pole output, inverted output disabled )
                self._write_reg(self._REG['MODE2'], 0x04)
                # MODE1レジスタ設定 ( Sleepモード解除, Auto-Increment有効, ALLCALL有効 )
                mode1 = self._read_reg(self._REG['MODE1'])
                mode1 = (mode1 & ~self._MODE1_FLAGS['SLEEP']) | self._MODE1_FLAGS['AI'] | self._MODE1_FLAGS['ALLCALL']
                self._write_reg(self._REG['MODE1'], mode1)
                time.sleep(0.005) # 安定化のための待機
                logger.info("PCA9685初期化成功")
                return
            except Exception as e:
                logger.error(f"PCA9685初期化試行 {attempt+1}/{Config.RETRIES} 失敗: {e}")
                if attempt == Config.RETRIES - 1:
                    raise ServoError("PCA9685初期化失敗") # 最終試行で失敗したら例外を発生
                time.sleep(Config.RETRY_DELAY) # 再試行前の待機

    def _write_reg(self, reg, value):
        """I2Cレジスタへの書き込みヘルパー関数"""
        try:
            buffer = bytes([reg & 0xFF]) + (value if isinstance(value, bytes) else bytes([value & 0xFF]))
            self.i2c.writeto(self.address, buffer)
            logger.debug(f"I2C書き込み: reg=0x{reg:02X}, value={value}")
        except Exception as e:
            logger.error(f"I2C書き込みエラー: reg=0x{reg:02X}, value={value}, error={e}")
            raise ServoError(f"I2C書き込みエラー: {e}")

    def _read_reg(self, reg):
        """I2Cレジスタからの読み込みヘルパー関数"""
        try:
            self.i2c.writeto(self.address, bytes([reg & 0xFF]))
            result = self.i2c.readfrom(self.address, 1)[0]
            logger.debug(f"I2C読み込み: reg=0x{reg:02X}, result=0x{result:02X}")
            return result
        except Exception as e:
            logger.error(f"I2C読み込みエラー: reg=0x{reg:02X}, error={e}")
            # エラー時でも続行できるよう、MODE1の場合はデフォルト値を返す
            return self._MODE1_FLAGS['AI'] | self._MODE1_FLAGS['ALLCALL'] if reg == self._REG['MODE1'] else 0

    def set_pwm_freq(self, freq_hz):
        """PWM周波数を設定します。"""
        try:
            # プリスケール値を計算
            prescale = max(3, min(255, int(round(self.osc_freq / (4096.0 * freq_hz) - 1))))
            old_mode = self._read_reg(self._REG['MODE1'])
            # Sleepモードに移行し、プリスケール値を設定
            self._write_reg(self._REG['MODE1'], (old_mode & ~self._MODE1_FLAGS['RESTART']) | self._MODE1_FLAGS['SLEEP'])
            self._write_reg(self._REG['PRESCALE'], prescale)
            # Sleepモード解除、再起動
            self._write_reg(self._REG['MODE1'], old_mode | self._MODE1_FLAGS['RESTART'])
            time.sleep(0.005)
            self._write_reg(self._REG['MODE1'], old_mode | self._MODE1_FLAGS['RESTART'])
            time.sleep(0.005)
            logger.info(f"PWM周波数設定成功: {freq_hz}Hz, prescale={prescale}")
        except Exception as e:
            logger.error(f"PWM周波数設定エラー: freq_hz={freq_hz}, error={e}")
            raise ServoError(f"PWM周波数設定エラー: {e}")

    def set_servo_pulse(self, channel, pulse_us, freq_hz=Config.PWM_FREQ_HZ):
        """
        指定されたチャンネルのサーボにパルス幅を設定します。
        Args:
            channel (int): サーボチャンネル (0-15)。
            pulse_us (float): パルス幅（マイクロ秒）。
            freq_hz (int): PWM周波数（Hz）。
        """
        try:
            # パルス幅をPCA9685の4096ステップに変換
            ticks = max(0, min(4095, int(round(pulse_us * 4096.0 / (1000000.0 / freq_hz)))))
            reg = self._REG['LED0_ON_L'] + 4 * channel # 各チャンネルのPWM設定レジスタの開始アドレス
            # PWM ON/OFF時間を設定
            self._write_reg(reg, bytes([0, 0, ticks & 0xFF, (ticks >> 8) & 0xFF]))
            logger.debug(f"サーボパルス設定: Channel={channel}, Pulse_us={pulse_us}, Ticks={ticks}")
        except Exception as e:
            logger.error(f"サーボパルス設定エラー: channel={channel}, pulse_us={pulse_us}, error={e}")
            raise ServoError(f"サーボパルス設定エラー: {e}")

    def set_servo_angle(self, channel, angle):
        """
        指定されたチャンネルのサーボを特定の角度に設定します。
        Args:
            channel (int): サーボチャンネル (0-15)。
            angle (float): 目標角度（度）。
        """
        # 角度範囲の定義
        angle_min = Config.ANGLE_CENTER - (Config.PAN_ANGLE_RANGE if channel == Config.CHANNELS[0] else Config.TILT_ANGLE_RANGE)
        angle_max = Config.ANGLE_CENTER + (Config.PAN_ANGLE_RANGE if channel == Config.CHANNELS[0] else Config.TILT_ANGLE_RANGE)
        
        # 角度を許容範囲内にクランプ
        clamped_angle = max(angle_min, min(angle_max, angle))
        if not angle_min <= angle <= angle_max:
            logger.warning(f"チャンネル{channel}の角度{angle:.2f}°が範囲外({angle_min:.2f}-{angle_max:.2f}°)。クランプ済: {clamped_angle:.2f}°")
        
        # 角度からパルス幅を計算 (0度=MIN_PULSE_US, 180度=MAX_PULSE_USと仮定)
        # 多くのサーボは0-180度で動作しますが、実際のサーボの仕様に合わせて調整が必要です。
        pulse_us = Config.MIN_PULSE_US + (Config.MAX_PULSE_US - Config.MIN_PULSE_US) * (clamped_angle / 180.0)
        self.set_servo_pulse(channel, pulse_us)
        logger.debug(f"サーボ角度設定: Channel={channel}, Angle={clamped_angle:.2f}° (元:{angle:.2f}°) -> Pulse_us={pulse_us:.2f}")

class ServoMover:
    """サーボを滑らかに動かすためのクラス"""
    def __init__(self, pca):
        self.pca = pca

    def _ease_in_out_quad(self, t):
        """イーズイン・イーズアウト（二次関数）の補間関数"""
        t *= 2
        return 0.5 * t * t if t < 1 else -0.5 * ((t - 1) * (t - 3) - 1)

    def move_smooth(self, channel, target_angle, current_angle):
        """
        指定されたチャンネルのサーボを現在の角度から目標角度まで滑らかに移動させます。
        Args:
            channel (int): サーボチャンネル。
            target_angle (float): 目標角度。
            current_angle (float): 現在の角度。
        Returns:
            float: 移動後の最終角度。
        """
        # 角度範囲の定義
        angle_min = Config.ANGLE_CENTER - (Config.PAN_ANGLE_RANGE if channel == Config.CHANNELS[0] else Config.TILT_ANGLE_RANGE)
        angle_max = Config.ANGLE_CENTER + (Config.PAN_ANGLE_RANGE if channel == Config.CHANNELS[0] else Config.TILT_ANGLE_RANGE)
        
        # 目標角度を範囲内にクランプ
        target_angle = max(angle_min, min(angle_max, target_angle))
        
        angle_diff = target_angle - current_angle
        if abs(angle_diff) < Config.MIN_ANGLE_DIFF:
            logger.debug(f"チャンネル {channel} 角度差が小さすぎる: {angle_diff:.2f}°。移動せず。")
            return current_angle # 角度差が小さい場合は移動しない

        step_delay = Config.MOVE_DURATION / Config.MOVE_STEPS # 各ステップ間の遅延時間
        
        try:
            for i in range(Config.MOVE_STEPS + 1):
                t = i / Config.MOVE_STEPS
                eased_t = self._ease_in_out_quad(t) # イージング関数を適用
                angle = current_angle + angle_diff * eased_t # 現在の角度から目標角度へ補間
                self.pca.set_servo_angle(channel, angle)
                time.sleep(step_delay)
            logger.info(f"チャンネル {channel} スムーズ移動完了: {target_angle:.2f}°（{'パン' if channel == Config.CHANNELS[0] else 'チルト'}）")
            return target_angle
        except ServoError as e:
            logger.error(f"チャンネル {channel} 移動エラー（ServoError）: {e}")
            raise # サーボエラーは再発生させる
        except Exception as e:
            logger.error(f"チャンネル {channel} 移動エラー（その他）: {e}")
            raise ServoError(f"チャンネル {channel} 移動失敗: {e}") # その他のエラーもServoErrorとして扱う

class UIBuilder:
    """ユーザーインターフェースを構築するクラス"""
    def __init__(self, controller):
        self.controller = controller

    def build(self):
        """UI要素を構築し、VBoxとして返します。"""
        ui_elements = []
        self.controller.debug_label = Label(value="中心差: X=0.0px (0.0°), Y=0.0px (0.0°)")
        ui_elements.append(self.controller.debug_label)
        # 変更点: error_labelの表示を削除
        # self.controller.error_label = Label(value="エラー: X=0.0px, Y=0.0px")
        # ui_elements.append(self.controller.error_label)

        # 検出対象選択ドロップダウンの追加
        self.controller.target_object_dropdown = Dropdown(
            options={name: class_id for name, class_id in Config.TARGET_OBJECTS.items()},
            value=Config.DEFAULT_TRACK_CLASS_ID,
            description='検出対象:'
        )
        self.controller.target_object_dropdown.observe(self.controller.set_target_object, names='value')
        ui_elements.append(self.controller.target_object_dropdown)
        
        # 各サーボチャンネルのスライダーとリセットボタン
        for channel in Config.CHANNELS: 
            angle_min = Config.ANGLE_CENTER - (Config.PAN_ANGLE_RANGE if channel == Config.CHANNELS[0] else Config.TILT_ANGLE_RANGE)
            angle_max = Config.ANGLE_CENTER + (Config.PAN_ANGLE_RANGE if channel == Config.CHANNELS[0] else Config.TILT_ANGLE_RANGE)
            slider = IntSlider(
                min=angle_min,
                max=angle_max,
                step=1,
                value=Config.ANGLE_CENTER,
                description=f"チャンネル{channel}角度({'パン' if channel == Config.CHANNELS[0] else 'チルト'})"
            )
            reset_button = Button(description=f"{Config.ANGLE_CENTER}°にリセット")
            slider.observe(self.controller.handle_angle_change(channel), names='value')
            reset_button.on_click(self.controller.handle_reset(channel))
            self.controller.sliders[channel] = slider
            ui_elements.append(HBox([slider, reset_button]))

        # 新機能: 位置記憶ボタンと移動ボタン
        self.controller.store_pos_button = Button(description="現在の位置を記憶")
        self.controller.store_pos_button.on_click(self.controller.store_current_angles)
        ui_elements.append(self.controller.store_pos_button)

        self.controller.move_to_stored_button = Button(description="記憶した位置へ移動")
        self.controller.move_to_stored_button.on_click(self.controller.move_to_stored_angles_handler)
        ui_elements.append(self.controller.move_to_stored_button)
        
        # 追尾開始/停止ボタン
        self.controller.tracking_button = Button(description="追尾開始")
        self.controller.tracking_button.on_click(self.controller.toggle_tracking)
        ui_elements.append(self.controller.tracking_button)
        
        # プログラム終了ボタン
        self.controller.exit_button = Button(description="プログラム終了")
        self.controller.exit_button.on_click(self.controller.handle_exit)
        ui_elements.append(self.controller.exit_button)
        
        # カメラ映像表示ウィジェット
        self.controller.image_widget = Image(width=Config.FRAME_WIDTH, height=Config.FRAME_HEIGHT)
        ui_elements.append(self.controller.image_widget)
        
        return VBox(ui_elements)

class ServoController:
    """サーボとカメラ、UIを統合して制御するメインクラス"""
    def __init__(self):
        self.pca = None
        self.mover = None
        self.i2c_bus = None
        self.sliders = {}
        self.image_widget = None
        self.tracking_button = None
        self.exit_button = None
        self.debug_label = None
        # 変更点: error_labelを削除
        # self.error_label = None 
        self.target_object_dropdown = None # ドロップダウンウィジェットへの参照
        self.cap = None
        self.yolo_model = None # YOLOモデルのインスタンス
        self.tracking = False # 追尾状態フラグ
        self.running = True # プログラム実行状態フラグ
        self.current_angles = {ch: Config.ANGLE_CENTER for ch in Config.CHANNELS} # 現在のサーボ角度
        self.target_class_id = Config.DEFAULT_TRACK_CLASS_ID # 追尾対象のクラスID
        self.last_detected_center = None # 最後に検出されたオブジェクトの中心座標 (x, y)

        # 新機能: 記憶された角度
        self.stored_angles = {ch: Config.ANGLE_CENTER for ch in Config.CHANNELS} # 記憶された角度を初期化

        # 新機能: 動画録画関連の変数
        # バッファの最大サイズは、バッファする秒数 * FPS
        self.frame_buffer = collections.deque(maxlen=Config.BUFFER_DURATION_SEC * Config.VIDEO_FPS) 
        self.is_buffering = False # バッファリング中かどうかのフラグ
        self.is_recording = False # 録画中かどうかのフラグ
        self.video_writer = None # cv2.VideoWriter オブジェクト
        self.tracking_start_time = 0.0 # 追尾開始時刻
        self.last_detection_time = 0.0 # 最後にオブジェクトが検出された時刻
        self.recording_start_time = 0.0 # 録画開始時刻
        self.consecutive_fail_count = 0 # 連続検出失敗回数

        # パンとチルト用のPIDコントローラを初期化
        self.pid_pan = PIDController(Config.KP_PAN, Config.KI_PAN, Config.KD_PAN, dt=1.0/Config.TRACKING_FPS)
        self.pid_tilt = PIDController(Config.KP_TILT, Config.KI_TILT, Config.KD_TILT, dt=1.0/Config.TRACKING_FPS)

        self._initialize()

    def _initialize(self):
        """コントローラ全体の初期化処理"""
        logger.setLevel(Config.LOG_LEVEL)
        try:
            logger.info("コントローラ初期化開始")
            
            # 動画保存ディレクトリが存在しない場合は作成
            if not os.path.exists(Config.VIDEO_SAVE_DIR):
                os.makedirs(Config.VIDEO_SAVE_DIR)
                logger.info(f"動画保存ディレクトリを作成しました: {Config.VIDEO_SAVE_DIR}")

            # I2Cバスの初期化
            try:
                self.i2c_bus = I2C(driver="ch341") # CH341 USB-I2Cアダプターを使用
                logger.info("CH341デバイス初期化成功")
            except Exception as e:
                logger.critical(f"CH341デバイス初期化失敗: {e}")
                raise ServoError(f"CH341デバイス初期化失敗: {e}")
            
            # PCA9685 PWMコントローラの初期化
            self.pca = PCA9685(self.i2c_bus)
            self.pca.set_pwm_freq(Config.PWM_FREQ_HZ)
            self.mover = ServoMover(self.pca)
            
            # サーボを初期角度（中央）に設定
            for channel in Config.CHANNELS:
                self.pca.set_servo_angle(channel, self.current_angles[channel])
                logger.info(f"チャンネル {channel} 初期角度設定: {self.current_angles[channel]}°")
            
            # カメラの初期化
            self.cap = cv2.VideoCapture(Config.CAMERA_ID)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) # 最新のフレームのみを読み込む設定
            if not self.cap.isOpened():
                raise ServoError(f"カメラ接続失敗: ID {Config.CAMERA_ID} が見つかりません。")
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.FRAME_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.FRAME_HEIGHT)
            logger.info(f"カメラ初期化成功: {Config.FRAME_WIDTH}x{Config.FRAME_HEIGHT}")
            
            # YOLOモデルのロード (初回実行時にモデルファイルがダウンロードされます)
            self.yolo_model = YOLO(Config.YOLO_MODEL_PATH)
            logger.info(f"YOLOv8モデルロード成功: {Config.YOLO_MODEL_PATH}")
            
            # UIの構築と表示
            ui_builder = UIBuilder(self)
            display(ui_builder.build())
            
            # 追尾処理を別スレッドで開始 (メインスレッドをブロックしないため)
            threading.Thread(target=self._track_objects, daemon=True).start()
            logger.info("追尾スレッド開始")

        except Exception as e:
            logger.critical(f"コントローラ初期化失敗: {e}", exc_info=True) # スタックトレースも出力
            self._cleanup() # エラー時はリソースを解放
            raise ServoError(f"コントローラ初期化失敗: {e}")

    def _cleanup(self):
        """プログラム終了時のリソース解放処理"""
        try:
            self.running = False # 追尾スレッドを停止させる
            if self.cap:
                self.cap.release() # カメラを解放
                logger.info("カメラリソース解放完了")
            
            # 録画中であれば停止し、ファイルをクローズ
            self._stop_recording_and_reset("クリーンアップ") 

            logger.info("YOLOモデルリソース解放完了 (もしあれば)")
            cv2.destroyAllWindows() # OpenCVのウィンドウを閉じる
            logger.info("すべてのリソース解放完了")
        except Exception as e:
            logger.error(f"リソース解放エラー: {e}")

    def store_current_angles(self, _):
        """現在のパン/チルト角度を記憶します。"""
        self.stored_angles[Config.CHANNELS[0]] = self.current_angles[Config.CHANNELS[0]]
        self.stored_angles[Config.CHANNELS[1]] = self.current_angles[Config.CHANNELS[1]]
        logger.info(f"現在の角度を記憶しました: パン={self.stored_angles[Config.CHANNELS[0]]:.1f}°, チルト={self.stored_angles[Config.CHANNELS[1]]:.1f}°")
        self.debug_label.value = f"角度記憶: パン={self.stored_angles[Config.CHANNELS[0]]:.1f}°, チルト={self.stored_angles[Config.CHANNELS[1]]:.1f}°"

    def move_to_stored_angles_handler(self, _):
        """記憶したパン/チルト角度へ移動します。"""
        if self.tracking:
            logger.warning("追尾中は記憶位置への移動はできません。")
            self.debug_label.value = "追尾中は移動できません。"
            return
        
        logger.info(f"記憶した位置へ移動開始: パン={self.stored_angles[Config.CHANNELS[0]]:.1f}°, チルト={self.stored_angles[Config.CHANNELS[1]]:.1f}°")
        try:
            # パンの移動
            self.current_angles[Config.CHANNELS[0]] = self.mover.move_smooth(
                Config.CHANNELS[0], self.stored_angles[Config.CHANNELS[0]], self.current_angles[Config.CHANNELS[0]]
            )
            self.sliders[Config.CHANNELS[0]].value = int(round(self.current_angles[Config.CHANNELS[0]]))
            
            # チルトの移動
            self.current_angles[Config.CHANNELS[1]] = self.mover.move_smooth(
                Config.CHANNELS[1], self.stored_angles[Config.CHANNELS[1]], self.current_angles[Config.CHANNELS[1]]
            )
            self.sliders[Config.CHANNELS[1]].value = int(round(self.current_angles[Config.CHANNELS[1]]))
            logger.info("記憶した位置への移動完了。")
            self.debug_label.value = "記憶位置へ移動完了。"
        except ServoError as e:
            logger.error(f"記憶位置への移動エラー: {e}")
            self.debug_label.value = f"移動エラー: {e}"

    def set_target_object(self, change):
        """UIのドロップダウンから選択された検出対象クラスIDを更新するハンドラ"""
        self.target_class_id = change['new']
        selected_name = next(name for name, id_val in Config.TARGET_OBJECTS.items() if id_val == self.target_class_id)
        logger.info(f"検出対象が変更されました: {selected_name} (クラスID: {self.target_class_id})")
        if self.tracking:
            self.pid_pan.reset()
            self.pid_tilt.reset()
            logger.info("検出対象変更に伴いPIDコントローラをリセット")
        self.last_detected_center = None # 対象変更時は近傍優先をリセット

    def handle_angle_change(self, channel):
        """手動スライダー操作時のサーボ角度変更ハンドラ"""
        def handler(change):
            if not self.tracking:
                try:
                    logger.info(f"手動操作: チャンネル {channel} を {change['new']}° に移動開始")
                    self.current_angles[channel] = self.mover.move_smooth(
                        channel, change['new'], self.current_angles[channel]
                    )
                    logger.info(f"手動操作: チャンネル {channel} 移動完了: {self.current_angles[channel]:.1f}°")
                except ServoError as e:
                    logger.error(f"手動角度変更エラー: {e}")
                    self.debug_label.value = f"手動操作エラー: {e}"
        return handler

    def handle_reset(self, channel):
        """リセットボタン押下時のサーボ角度リセットハンドラ"""
        def handler(_):
            if not self.tracking:
                try:
                    logger.info(f"リセット操作: チャンネル {channel} を {Config.ANGLE_CENTER}° に移動開始")
                    self.current_angles[channel] = self.mover.move_smooth(
                        channel, Config.ANGLE_CENTER, self.current_angles[channel]
                    )
                    self.sliders[channel].value = Config.ANGLE_CENTER
                    logger.info(f"リセット操作: チャンネル {channel} 移動完了: {self.current_angles[channel]:.1f}°")
                except ServoError as e:
                    logger.error(f"リセットエラー: {e}")
                    self.debug_label.value = f"リセットエラー: {e}"
        return handler

    def toggle_tracking(self, _):
        """追尾開始/停止ボタンのハンドラ"""
        self.tracking = not self.tracking
        self.tracking_button.description = "追尾停止" if self.tracking else "追尾開始"
        if self.tracking:
            logger.info("追尾開始")
            self.pid_pan.reset()
            self.pid_tilt.reset()
            self.last_detected_center = None # 追尾開始時は近傍優先をリセット
            # 変更点: ここでの_start_buffering()の呼び出しを削除
            # self._start_buffering() # 追尾開始時にバッファリングも開始
        else:
            logger.info("追尾停止")
            self._stop_recording_and_reset("追尾停止") # 追尾停止時に録画/バッファリングも停止
            # 変更点: 追尾停止時にis_bufferingもリセット
            self.is_buffering = False
            self.frame_buffer.clear()
            try:
                # 追尾停止時、サーボを中央に戻す
                for channel in Config.CHANNELS:
                    logger.info(f"追尾停止: チャンネル {channel} を中心 {Config.ANGLE_CENTER}° にリセット開始")
                    self.current_angles[channel] = self.mover.move_smooth(
                        channel, Config.ANGLE_CENTER, self.current_angles[channel]
                    )
                    self.sliders[channel].value = Config.ANGLE_CENTER
                    logger.info(f"追尾停止: チャンネル {channel} リセット完了: {self.current_angles[channel]:.1f}°")
            except ServoError as e:
                logger.error(f"追尾停止リセットエラー: {e}")
                self.debug_label.value = f"追尾停止エラー: {e}"
            self.debug_label.value = "中心差: X=0.0px (0.0°), Y=0.0px (0.0°)"

    def handle_exit(self, _):
        """プログラム終了ボタンのハンドラ"""
        logger.info("プログラム終了ボタンが押されました")
        self._cleanup() # リソース解放
        self.debug_label.value = "プログラム終了"
        self.tracking_button.disabled = True
        self.exit_button.disabled = True
        for slider in self.sliders.values():
            slider.disabled = True

    def _start_buffering(self):
        """フレームバッファリングを開始します。"""
        self.frame_buffer.clear()
        self.is_buffering = True
        self.tracking_start_time = time.time() # 追尾開始時刻 (バッファリング開始時刻)
        self.last_detection_time = time.time() # 追尾開始時は検出があったとみなす (バッファリング開始時刻)
        self.consecutive_fail_count = 0
        logger.info("フレームバッファリング開始。")
        self.debug_label.value = "バッファリング中..."

    def _start_recording(self):
        """録画を開始します。バッファ内のフレームも書き込みます。"""
        if self.video_writer is not None and self.video_writer.isOpened():
            logger.warning("既に録画中です。新たな録画を開始する前に既存の録画を停止します。")
            self._stop_recording_and_reset("多重録画開始") # 既に録画中なら一度停止

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{Config.VIDEO_FILENAME_PREFIX}{timestamp}.avi"
        filepath = os.path.join(Config.VIDEO_SAVE_DIR, filename) # 保存ディレクトリを指定

        self.video_writer = cv2.VideoWriter(
            filepath, Config.VIDEO_CODEC, Config.VIDEO_FPS,
            (Config.FRAME_WIDTH, Config.FRAME_HEIGHT)
        )
        if not self.video_writer.isOpened():
            logger.error(f"ビデオファイル {filepath} のオープンに失敗しました。コーデックやパスを確認してください。")
            self.is_recording = False
            return

        # バッファ内のフレームを書き出す (プリロール)
        logger.info(f"バッファから{len(self.frame_buffer)}フレームを書き込み中...")
        for buffered_frame in self.frame_buffer:
            self.video_writer.write(buffered_frame)
        self.frame_buffer.clear() # 書き込んだらバッファをクリア

        self.is_recording = True
        self.recording_start_time = time.time()
        logger.info(f"録画開始: {filepath}")
        self.debug_label.value = f"録画中: {filename}"

    def _stop_recording_and_reset(self, reason="不明"):
        """録画を停止し、リソースを解放し、バッファをクリアします。"""
        if self.is_recording and self.video_writer and self.video_writer.isOpened():
            self.video_writer.release()
            self.video_writer = None
            self.is_recording = False
            self.is_buffering = False # 明示的にバッファリングも停止
            self.frame_buffer.clear() # バッファをクリア
            logger.info(f"録画停止: {reason}。ビデオファイルをクローズし、バッファをクリアしました。")
            self.debug_label.value = f"録画停止: {reason}"
            # 録画停止後、記憶した位置へ移動 (追尾が停止している場合のみ)
            # スレッドセーフティのため、別スレッドで実行
            threading.Thread(target=self._move_to_stored_angles_after_recording, daemon=True).start()
        elif self.is_buffering: # 録画はしてないがバッファリング中の場合
            self.is_buffering = False
            self.frame_buffer.clear()
            logger.info(f"バッファリング停止: {reason}。バッファをクリアしました。")
            self.debug_label.value = f"バッファリング停止: {reason}"

    def _move_to_stored_angles_after_recording(self):
        """録画停止後に記憶した位置へ移動するヘルパー関数"""
        # 追尾が停止していることを確認してから移動
        # ここでは追尾状態を制御しているため、_track_objectsループ外で実行
        if not self.tracking: # 追尾ボタンが「追尾開始」になっていることを確認
            logger.info("録画停止後、記憶位置への移動を開始します。")
            self.move_to_stored_angles_handler(None) # Noneはダミーのイベントオブジェクト
        else:
            logger.warning("追尾中のため、録画停止後の記憶位置への移動はスキップされました。")

    def _track_objects(self):
        """
        物体追尾のメインループ。
        カメラからフレームを読み込み、物体検出を行い、サーボを制御します。
        """
        frame_interval = 1.0 / Config.TRACKING_FPS # 目標フレーム間隔
        last_frame_time = time.time() # 前回のフレーム処理開始時刻
        last_display_time = time.time() # 前回の画面表示時刻

        while self.running and self.cap.isOpened():
            current_time = time.time()
            # フレームレートを制御するための待機
            if current_time - last_frame_time < frame_interval:
                time.sleep(frame_interval - (current_time - last_frame_time))
                current_time = time.time() # スリープ後の正確な時刻を再取得
            last_frame_time = current_time

            ret, frame = self.cap.read() # カメラからフレームを読み込み
            if not ret:
                logger.warning("カメラフレーム読み込み失敗。")
                self.debug_label.value = "中心差: カメラエラー"
                continue

            try:
                # YOLOv8で物体を検出
                # verbose=FalseでYOLOのログ出力を抑制
                # conf=Config.CONF_THRESHOLDで検出信頼度閾値を設定
                results = self.yolo_model(frame, conf=Config.CONF_THRESHOLD, verbose=False)

                selected_object_center = None
                best_bbox = None
                best_conf = -1
                
                # 検出結果がある場合のみ処理
                if results and results[0].boxes:
                    candidate_detections = []
                    
                    # Step 1: 前回検出したオブジェクトがある場合、その近傍 (ROI) を優先して検索
                    if self.last_detected_center:
                        roi_x_min = max(0, self.last_detected_center[0] - Config.SEARCH_RADIUS_PX)
                        roi_y_min = max(0, self.last_detected_center[1] - Config.SEARCH_RADIUS_PX)
                        roi_x_max = min(Config.FRAME_WIDTH, self.last_detected_center[0] + Config.SEARCH_RADIUS_PX)
                        roi_y_max = min(Config.FRAME_HEIGHT, self.last_detected_center[1] + Config.SEARCH_RADIUS_PX)

                        # ROIの矩形をフレームに描画 (デバッグ用)
                        cv2.rectangle(frame, (roi_x_min, roi_y_min), (roi_x_max, roi_y_max), (255, 255, 0), 1) # 黄色

                        for box in results[0].boxes:
                            class_id = int(box.cls)
                            confidence = float(box.conf)
                            # 信頼度閾値と対象クラスIDに一致する検出のみを考慮
                            if class_id == self.target_class_id and confidence >= Config.CONF_THRESHOLD:
                                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                                obj_center_x = (x1 + x2) // 2
                                obj_center_y = (y1 + y2) // 2

                                # オブジェクトの中心がROI内にあるかチェック
                                if roi_x_min <= obj_center_x <= roi_x_max and \
                                   roi_y_min <= obj_center_y <= roi_y_max:
                                    candidate_detections.append((box, obj_center_x, obj_center_y, confidence))
                    
                    if candidate_detections:
                        # ROI内に候補が見つかった場合、前回検出した中心に最も近いものを選択
                        min_distance = float('inf')
                        closest_detection = None
                        for box, cx, cy, conf in candidate_detections:
                            dist = np.sqrt((cx - self.last_detected_center[0])**2 + (cy - self.last_detected_center[1])**2)
                            if dist < min_distance:
                                min_distance = dist
                                closest_detection = (box, cx, cy, conf)
                        
                        if closest_detection:
                            best_bbox = closest_detection[0].xyxy[0].cpu().numpy()
                            selected_object_center = (closest_detection[1], closest_detection[2])
                            best_conf = closest_detection[3]
                            class_name = [name for name, id_val in Config.TARGET_OBJECTS.items() if id_val == self.target_class_id][0]
                            logger.debug(f"優先: ROI内で対象({class_name})を検出。距離={min_distance:.1f}px, 信頼度={best_conf:.2f}")

                    # Step 2: ROI内に見つからなかった場合、または前回検出オブジェクトがない場合、全フレームを検索
                    if selected_object_center is None:
                        for box in results[0].boxes:
                            class_id = int(box.cls)
                            confidence = float(box.conf)
                            # 指定されたクラスIDかつ、信頼度閾値以上の検出のみを考慮
                            if class_id == self.target_class_id and confidence >= Config.CONF_THRESHOLD:
                                if confidence > best_conf: # 最も信頼度が高いものを選択
                                    best_conf = confidence
                                    best_bbox = box.xyxy[0].cpu().numpy()
                                    x1, y1, x2, y2 = map(int, best_bbox)
                                    selected_object_center = ((x1 + x2) // 2, (y1 + y2) // 2)
                        if selected_object_center:
                            class_name = [name for name, id_val in Config.TARGET_OBJECTS.items() if id_val == self.target_class_id][0]
                            logger.debug(f"フォールバック: 全フレームから対象({class_name})を検出。信頼度={best_conf:.2f}")

                # 最終的に追尾対象が選択された場合
                if selected_object_center:
                    # 変更点: オブジェクトが検出されたら、バッファリングが開始されていなければ開始
                    if self.tracking and not self.is_buffering:
                        self._start_buffering()

                    # 最後に検出された中心座標を更新
                    self.last_detected_center = selected_object_center
                    obj_center_x, obj_center_y = selected_object_center

                    # 検出されたオブジェクトをフレームに描画
                    if best_bbox is not None:
                        x1, y1, x2, y2 = map(int, best_bbox)
                        
                        # 変更点: バッファリング中であればバウンディングボックスの色を赤に
                        bbox_color = (0, 0, 255) if self.is_buffering else (0, 255, 0) # 赤 or 緑

                        cv2.rectangle(frame, (x1, y1), (x2, y2), bbox_color, 2) # 矩形の色を動的に変更
                        cv2.circle(frame, (obj_center_x, obj_center_y), 5, (0, 0, 255), -1) # 赤色の中心点
                        class_name = [name for name, id_val in Config.TARGET_OBJECTS.items() if id_val == self.target_class_id][0]
                        cv2.putText(frame, f"{class_name} ({best_conf:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bbox_color, 2) # テキストの色も動的に変更


                    frame_center_x = Config.FRAME_WIDTH // 2
                    frame_center_y = Config.FRAME_HEIGHT // 2
                    
                    # ピクセル誤差を計算 (画面中心からのずれ)
                    error_px_x = obj_center_x - frame_center_x
                    error_px_y = frame_center_y - obj_center_y if not Config.DEBUG_FIX_TILT_ZERO else 0

                    # デッドバンドの適用 (現在の角度からの偏差ではなく、固定値を使用)
                    deadband_x = Config.DEADBAND_PX_BASE
                    deadband_y = Config.DEADBAND_PX_BASE

                    current_error_x_px = error_px_x # デバッグ表示用の生の誤差
                    current_error_y_px = error_px_y

                    # デッドバンド内であれば誤差をゼロとみなす
                    if abs(error_px_x) <= deadband_x:
                        error_px_x = 0
                    if abs(error_px_y) <= deadband_y:
                        error_px_y = 0

                    # ピクセル誤差を角度誤差に変換 (PIDコントローラへの入力)
                    # カメラのFOVとフレームサイズから、1ピクセルあたりの角度を計算
                    angle_per_px_x = Config.CAMERA_FOV_X / Config.FRAME_WIDTH
                    angle_per_px_y = Config.CAMERA_FOV_Y / Config.FRAME_HEIGHT

                    angle_error_x = error_px_x * angle_per_px_x
                    angle_error_y = error_px_y * angle_per_px_y
                    
                    logger.debug(f"Px Error: X={current_error_x_px:.1f}, Y={current_error_y_px:.1f} | Angle Error: X={angle_error_x:.2f}°, Y={angle_error_y:.2f}° | Deadband: X={deadband_x:.1f}px, Y={deadband_y:.1f}px")

                    if self.tracking:
                        pan_channel, tilt_channel = Config.CHANNELS[0], Config.CHANNELS[1]

                        # PIDコントローラで角度変化量を計算
                        delta_angle_pan = self.pid_pan.update(angle_error_x)
                        delta_angle_tilt = self.pid_tilt.update(angle_error_y)
                        
                        # 目標角度を計算 (パンは誤差の符号によって逆方向に動くことが多い)
                        target_pan = self.current_angles[pan_channel] - delta_angle_pan
                        target_tilt = self.current_angles[tilt_channel] + delta_angle_tilt

                        logger.debug(f"Tracking: Current Pan={self.current_angles[pan_channel]:.1f}°, Delta Pan={delta_angle_pan:.2f}°, Target Pan={target_pan:.1f}°")
                        logger.debug(f"Tracking: Current Tilt={self.current_angles[tilt_channel]:.1f}°, Delta Tilt={delta_angle_tilt:.2f}°, Target Tilt={target_tilt:.1f}°")

                        try:
                            # パンの移動 (角度変化量が最小閾値より大きい場合のみ)
                            if abs(delta_angle_pan) > Config.MIN_ANGLE_DIFF:
                                self.current_angles[pan_channel] = self.mover.move_smooth(
                                    pan_channel, target_pan, self.current_angles[pan_channel]
                                )
                                self.sliders[pan_channel].value = int(round(self.current_angles[pan_channel]))

                            # チルトの移動 (角度変化量が最小閾値より大きい場合のみ)
                            if abs(delta_angle_tilt) > Config.MIN_ANGLE_DIFF:
                                self.current_angles[tilt_channel] = self.mover.move_smooth(
                                    tilt_channel, target_tilt, self.current_angles[tilt_channel]
                                )
                                self.sliders[tilt_channel].value = int(round(self.current_angles[tilt_channel]))
                        except ServoError as e:
                            logger.error(f"サーボ移動中にエラーが発生: {e}")
                            self.debug_label.value = f"サーボ移動エラー: {e}"
                            time.sleep(Config.RETRY_DELAY * 2)

                    # デバッグ用ラベルを更新
                    self.debug_label.value = (
                        f"中心差: X={current_error_x_px:.1f}px ({angle_error_x:.1f}°), Y={current_error_y_px:.1f}px ({angle_error_y:.1f}°)"
                    )

                else:
                    # 指定オブジェクトが検出されなかった場合
                    self.debug_label.value = "Debug: 指定オブジェクト検出なし"
                    self.last_detected_center = None # 検出できなかった場合は近傍優先をリセット
                    if self.tracking:
                        # 追尾中に検出が途切れたらPIDコントローラをリセット
                        self.pid_pan.reset()
                        self.pid_tilt.reset()
                        logger.debug("指定オブジェクト未検出のためPIDコントローラをリセット")

                # フレームバッファリングと録画のロジック
                if self.tracking:
                    current_time = time.time()
                    
                    # フレームをバッファに追加 (録画中であってもバッファリングは継続)
                    # 変更点: バッファリングが開始されている場合のみフレームを追加
                    if self.is_buffering:
                        self.frame_buffer.append(frame.copy()) # フレームのコピーを保存

                    # 録画開始条件: 追尾開始からX秒経過 AND まだ録画中でない AND オブジェクトが検出されている
                    if not self.is_recording and self.is_buffering and \
                       selected_object_center is not None and \
                       (current_time - self.tracking_start_time >= Config.BUFFER_START_DELAY_SEC):
                        self._start_recording()

                    if self.is_recording:
                        # 現在のフレームを録画ファイルに書き込む
                        if self.video_writer and self.video_writer.isOpened():
                            self.video_writer.write(frame)
                        
                        # 録画時間Z秒経過で停止
                        if current_time - self.recording_start_time >= Config.MAX_RECORDING_DURATION_SEC:
                            self.tracking = False # 追尾を停止
                            self.tracking_button.description = "追尾開始"
                            self.debug_label.value = "最大録画時間経過により追尾停止"
                            self._stop_recording_and_reset("最大録画時間経過") # 録画停止とリセット
                            continue # 次のループへ

                        # 検出が連続Y回失敗で停止
                        if selected_object_center is None: # 今回のフレームでオブジェクトが検出されなかった場合
                            self.consecutive_fail_count += 1
                            logger.debug(f"検出失敗回数: {self.consecutive_fail_count}")
                            if self.consecutive_fail_count >= Config.DETECTION_FAIL_COUNT_THRESHOLD:
                                self.tracking = False # 追尾を停止
                                self.tracking_button.description = "追尾開始"
                                self.debug_label.value = "連続検出失敗により追尾停止"
                                self._stop_recording_and_reset("連続検出失敗") # 録画停止とリセット
                                continue # 次のループへ
                        else: # オブジェクトが検出された場合
                            self.consecutive_fail_count = 0 # 検出成功でリセット
                            self.last_detection_time = current_time # 最終検出時刻を更新
                else:
                    # 追尾が停止している場合、録画/バッファリングも停止していることを確認
                    if self.is_recording or self.is_buffering:
                        self._stop_recording_and_reset("追尾停止")


            except Exception as e:
                logger.error(f"追尾ループエラー: {e}", exc_info=True) # スタックトレースも出力
                self.debug_label.value = f"追尾エラー: {e}"

            # 画面中心の描画 (青色の点)
            frame_center_x = Config.FRAME_WIDTH // 2
            frame_center_y = Config.FRAME_HEIGHT // 2
            cv2.circle(frame, (frame_center_x, frame_center_y), 5, (255, 0, 0), -1)

            # フレームの表示頻度制御
            # カメラのFPSとは別に、UIの更新頻度を制限して負荷を軽減
            if time.time() - last_display_time >= 1.0 / 30: # 例: 30FPSで表示
                self._display_frame(frame)
                last_display_time = time.time()

    def _display_frame(self, frame):
        """OpenCVのフレームをJPEG形式にエンコードし、Imageウィジェットに表示します。"""
        _, buffer = cv2.imencode('.jpg', frame)
        self.image_widget.value = buffer.tobytes()

if __name__ == "__main__":
    try:
        controller = ServoController()
    except ServoError as e:
        print(f"致命的なエラーが発生しました: {e}")
