import uuid
import mss
import numpy as np
from gym.utils import seeding
from gym import error, spaces, utils
import gym
import yaml
from termcolor import cprint
import time
import threading
import subprocess
import os
import json
import itertools
import inspect
from contextlib import contextmanager
import array
from src.utils import next_free_port
import wandb
import abc
from enum import auto
from PIL import Image
from pathlib import Path
import sys
import socket

PY3_OR_LATER = sys.version_info[0] >= 3


###############################################

class ImageHelper:

    def GetPixelColor(self, image_array, x, y):
        base_pixel = image_array[y][x]
        red = base_pixel[0]
        green = base_pixel[1]
        blue = base_pixel[2]
        return (red, green, blue)


###############################################
### Variables & Constants                   ###
###############################################

# The width, height, and depth of the emulator window:
# SCR_W = 640
# SCR_H = 480
SCR_W = 320
SCR_H = 240
SCR_D = 3

COUNTPEROP = 1
MILLISECOND = 1.0 / 1000.0

DEFAULT_PORT = 8082

IMAGE_HELPER = ImageHelper()

BENCHMARK = False


###############################################
class Mupen64PlusEnv(gym.Env):
    __metaclass__ = abc.ABCMeta
    metadata = {'render.modes': ['human', 'rgb_array']}

    resolutions = {
        "normal": (640, 480),
        "small": (320, 240),
        # the logical next resolution would have been 160, 120. however the progress bar is not rendered properly in that resolution, so a slightly larger ones is picked instead
        "supersmall": (170, 128),
    }

    def __init__(self, use_wandb=True, run=None, input_port="8032", vnc_port="5009", benchmark=True, resolution="supersmall", containerized=True, quiet=False, res_w=None, res_h=None, auto_abort=True, variable_episode_length=False, base_episode_length=20000, episode_length_increase=1, gray_scale=True):
        global SCR_W, SCR_H
        if res_w is not None and res_h is not None:
            self.res_w = res_w
            self.res_h = res_h
        else:
            self.res_w, self.res_h = self.resolutions[resolution]
        cprint(f"using resolution {self.res_w}x{self.res_h}")
        SCR_W, SCR_H = self.res_w, self.res_h
        self.variable_episode_length = variable_episode_length
        self.episode_length = base_episode_length
        self.episode_length_increase = episode_length_increase
        self.quiet = quiet

        self.input_port = next_free_port(int(input_port), self.quiet)
        self.vnc_port = vnc_port
        self.viewer = None
        self.benchmark = benchmark
        self.reset_count = 0
        self.step_count = 0
        self.running = True
        self.episode_aborted = False
        self.episode_completed = False
        self.auto_abort = auto_abort
        self.episode_reward = 0
        self.last_episode_reward = 0
        self.max_duration = 0
        self.max_reward = 0
        self.total_progress = 0
        self.pixel_array = None
        self.container_name = ""
        self.gray_scale = gray_scale
        self.use_wandb = use_wandb
        self.run = run
        self._base_load_config()
        self._base_validate_config()
        self.frame_skip = self.config['FRAME_SKIP']
        if self.frame_skip < 1:
            self.frame_skip = 1

        self.controller_server = self._start_controller_server()

        initial_disp = os.environ["DISPLAY"]
        if not self.quiet:
            cprint('Initially on DISPLAY %s' % initial_disp, 'red')
        self.containerized = containerized
        if containerized:
            self.xvfb_process, self.emulator_process = self.start_container(
                rom_name=self.config['ROM_NAME'],
                gfx_plugin=self.config['GFX_PLUGIN'],
                input_driver_path=self.config['INPUT_DRIVER_PATH'],
                res_w=SCR_W, res_h=SCR_H, res_d=SCR_D,
                image=self.config["IMAGE_SPEC"],
            )
        else:
            self.xvfb_process, self.emulator_process = self._start_emulator(
                    rom_name=self.config['ROM_NAME'],
                    gfx_plugin=self.config['GFX_PLUGIN'],
                    input_driver_path=self.config['INPUT_DRIVER_PATH'],
                    res_w=SCR_W, res_h=SCR_H, res_d=SCR_D
            )
        time.sleep(2)

        # Restore the DISPLAY env var
        os.environ["DISPLAY"] = initial_disp
        if not self.quiet:
            cprint('Changed back to DISPLAY %s' % os.environ["DISPLAY"], 'red')

        with self.controller_server.frame_skip_disabled():
            self._navigate_menu()

        self.observation_space = \
            spaces.Box(low=0, high=255, shape=(
                SCR_H, SCR_W, 1), dtype=np.uint8)

        actions = [[-80, 80],  # Joystick X-axis
                   [-80, 80],  # Joystick Y-axis
                   [0,  1],  # A Button
                   [0,  1],  # B Button
                   [0,  1],  # RB Button
                   [0,  1],  # LB Button
                   [0,  1],  # Z Button
                   [0,  1],  # C Right Button
                   [0,  1],  # C Left Button
                   [0,  1],  # C Down Button
                   [0,  1],  # C Up Button
                   [0,  1],  # D-Pad Right Button
                   [0,  1],  # D-Pad Left Button
                   [0,  1],  # D-Pad Down Button
                   [0,  1],  # D-Pad Up Button
                   [0,  1],  # Start Button
                   ]

        self.action_space = spaces.MultiDiscrete(
            [len(action) for action in actions])

    def _base_load_config(self):
        self.config = yaml.safe_load(
            open(os.path.join(os.path.dirname(inspect.stack()[0][1]), "config.yml")))
        self._load_config()

    @abc.abstractmethod
    def _load_config(self):
        return

    def _base_validate_config(self):
        if 'ROM_NAME' not in self.config:
            raise AssertionError('ROM_NAME configuration is required')
        if 'GFX_PLUGIN' not in self.config:
            raise AssertionError('GFX_PLUGIN configuration is required')
        self._validate_config()

    @abc.abstractmethod
    def _validate_config(self):
        return

    def _step(self, action):
        image = self._act(action)
        obs = self._observe(image)
        if self.step_count >= self.episode_length:
            cprint("aborting episode due to max steps reached!", "cyan")
            self.episode_aborted = True
            self.episode_completed = False
        else:
            self.episode_completed, self.episode_aborted = self._evaluate_end_state()

        if not self.auto_abort:
            self.episode_aborted = False
        reward = self._get_reward()

        self.step_count += 1
        self.episode_reward += reward

        if self.gray_scale:
            obs = np.average(obs, axis=2, weights=[
                             0.299, 0.587, 0.114], keepdims=True).astype(np.uint8)
        if self.episode_aborted:
            if not self.quiet:
                cprint("Episode aborted!", "cyan")
        if self.episode_completed:
            if not self.quiet:
                cprint("Episode successfully completed!", "cyan")
            if self.use_wandb and wandb.run is not None:
                wandb.log({"env/episode-stop-reason": 3})
        return obs, reward, self.episode_aborted or self.episode_completed, {}

    def _act(self, action, count=1, force_count=False):
        if not self.controller_server.frame_skip_enabled and not force_count:
            for _ in itertools.repeat(None, count):
                image = self.controller_server.send_controls(ControllerState(action))
        else:
            image = self.controller_server.send_controls(ControllerState(
                action), count=count, force_count=force_count)
        return image

    def _wait(self, count=1, wait_for='Unknown'):
        self._act(ControllerState.NO_OP, count=count, force_count=True)

    def _press_button(self, button, times=1):
        for _ in itertools.repeat(None, times):
            self._act(button)  # Press
            self._act(ControllerState.NO_OP)  # and release

    def _observe(self, image=None):
        if image is None:
            image = self._act([0] * 16)
        # somehow the fb for the smallest resolution is only (170, 127) pixels large, so we have to append a line.
        # other resolutions are not affected by this bug.
        if self.res_w == 170:
            image += b'\xff' * 170 * 3

        self.pixel_array = np.frombuffer(image, dtype=np.uint8).reshape(self.res_h, self.res_w, 3)
        self.pixel_array = np.flip(self.pixel_array[:, :, :3], 2)
        return self.pixel_array

    @abc.abstractmethod
    def _navigate_menu(self):
        return

    @abc.abstractmethod
    def _get_reward(self):
        # cprint('Get Reward called!', 'yellow')
        return 0

    @abc.abstractmethod
    def _evaluate_end_state(self):
        # cprint('Evaluate End State called!', 'yellow')
        return False, False

    @abc.abstractmethod
    def _reset(self):
        if not self.quiet:
            cprint('Reset called!', 'yellow')
        self.reset_count += 1
        self.last_episode_reward = self.episode_reward

        self.max_reward = max(self.max_reward, self.episode_reward)
        self.max_duration = max(self.max_duration, self.step_count)
        # if not self.quiet:
        cprint(
        f"last episode reward: {self.episode_reward:.1f}, duration: {self.step_count}, progress: {self.total_progress}", "green")
        if self.use_wandb and self.run is not None:
            self.run.log({
                "env/rewards": self.episode_reward,
                "env/length": self.step_count,
                "env/max-reward": self.max_reward,
                "env/max-duration": self.max_duration,
            })
        self.episode_reward = 0
        if self.reset_count > 1 and self.variable_episode_length:
            self.episode_length += self.episode_length_increase
            if not self.quiet:
                cprint(f"next episode length: {self.episode_length}", "yellow")

        self.step_count = 0
        obs = self._observe()
        if self.gray_scale:
            obs = np.average(obs, axis=2, weights=[
                             0.299, 0.587, 0.114], keepdims=True).astype(np.uint8)
        return obs

    def _render(self, mode='human', close=False):
        if close:
            if hasattr(self, 'viewer') and self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
        img = self.pixel_array
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            if not hasattr(self, 'viewer') or self.viewer is None:
                from gym.envs.classic_control import rendering
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)

    def _close(self):
        cprint('Close called!', 'yellow')
        self.running = False
        self._kill_emulator()
        self._stop_controller_server()

    def _start_controller_server(self):
        if self.res_w == 170:
            num_pixels = self.res_w * (self.res_h - 1)
        else:
            num_pixels = self.res_w * self.res_h
        server = ControllerUpdater(
            input_host='',
            input_port=self.input_port,  # gets port for external
            control_timeout=self.config['ACTION_TIMEOUT'],
            frame_skip=self.frame_skip,
            num_pixels=num_pixels)  # TODO: Environment argument (with issue #26)
        if not self.quiet:
            print('ControllerUpdater started on port ', self.input_port)
        return server

    def _stop_controller_server(self):
        # cprint('Stop Controller Server called!', 'yellow')
        if hasattr(self, 'controller_server'):
            self.controller_server.shutdown()

    def start_container(self,
                        rom_name,
                        gfx_plugin,
                        input_driver_path,
                        image,
                        res_w=SCR_W,
                        res_h=SCR_H,
                        res_d=SCR_D):
        self.container_name = f"Mario-Kart-Env-Container-{str(uuid.uuid4())[:4]}"
        
        rom_path = os.path.abspath(
            os.path.join(os.path.dirname(inspect.stack()[0][1]),
                         '../ROMs',
                         rom_name))

        rom_dir = Path(rom_path).parent
        self.check_rom_path(rom_path)

        input_driver_path = os.path.abspath(
            os.path.expanduser(input_driver_path))
        if not os.path.isfile(input_driver_path):
            msg = "Input driver not found: " + input_driver_path
            cprint(msg, 'red')
            raise Exception(msg)

        benchmark_options = [
            "--nospeedlimit",
            "--set", f"Core[CountPerOp]={COUNTPEROP}",
            # "--set", f"Core[DelaySI]=False",
            # "--set", f"Video-Glide64[filtering]=0",
            # "--set", f"Video-Glide64[fast_crc]=True",
            # "--set", f"Video-Glide64[fb_hires]=False",
            # "--set", "Video-Rice[ScreenUpdateSetting]=3",
            # "--set", "Video-Rice[FastTextureLoading]=1",
            # "--set", "Video-Rice[TextureQuality]=2",
            # "--set", "Video-Rice[ColorQuality]=1",
        ]

        cmd = [self.config['MUPEN_CMD'],
               # "--nospeedlimit",
               "--nosaveoptions",
               "--resolution",
               "%ix%i" % (res_w, res_h),
               "--gfx", gfx_plugin,
               "--audio", "dummy",
               "--set", f"Input-Bot-Control0[port]={self.input_port}",
            #    "--input", input_driver_path,
               "--input", "/src/code/install/mupen64plus-input-bot/mupen64plus-input-bot.so",
               "/src/gym-mupen64plus/gym_mupen64plus/ROMs/" + Path(rom_path).name]

        if self.benchmark:
            cmd = [cmd[0]] + benchmark_options + cmd[1:]

        xvfb_cmd = ["docker",
                    "run", 
                    "--name",
                    self.container_name,
                    "-p",
                    str(self.input_port) + ":" + str(self.input_port),
                    # "-v",
                    # str(rom_dir.resolve()) + ":/src/gym-mupen64plus/gym_mupen64plus/ROMs",
                    "-v",
                    "/home/Paul.Mattes/Mario-Kart-RL:/src/code",
                    "-di",
                    image,
                    self.config['XVFB_CMD'],
                    ":1",
                    "-screen",
                    "0",
                    "%ix%ix%i" % (res_w, res_h, res_d * 8),
                    "-noreset",
                    "-fbdir",
                    self.config['TMP_DIR']]
        
        subprocess.run(xvfb_cmd, stderr=subprocess.STDOUT)
        cprint('Starting xvfb with command: %s' % " ".join(xvfb_cmd), 'yellow')
        time.sleep(3)  # Give xvfb a couple seconds to start up

        log_cmd = ["docker", "logs", self.container_name]
        print("quiet:", self.quiet)
        cprint("running logs ", "yellow")

        log_process = subprocess.run(log_cmd, stderr=subprocess.STDOUT)


        cmd = [
            "docker",
            "exec",
            "-te", "DISPLAY=:1",
            "-e", "XVFB_FB_PATH=" + self.config["TMP_DIR"] + "/Xvfb_screen0",
            self.container_name,
            self.config['VGLRUN_CMD'],
                "-d", ":1"] + cmd

        cprint('Starting emulator with comand: %s' % " ".join(cmd), 'yellow')

        # emulator_process = subprocess.Popen(cmd, stderr=subprocess.STDOUT, stdout=subprocess.DEVNULL if self.quiet else sys.stdout)
        emulator_process = subprocess.Popen(cmd, shell=False,
                                            stderr=subprocess.STDOUT)

        # emu_mon = EmulatorMonitor()
        # monitor_thread = threading.Thread(target=emu_mon.monitor_emulator,
        #                                   args=[emulator_process])
        # monitor_thread.daemon = True
        # monitor_thread.start()

        return None, emulator_process

    def check_rom_path(self, rom_path):
        if not os.path.isfile(rom_path):
            msg = "ROM not found: " + rom_path
            cprint(msg, 'red')
            rom_dir = Path(rom_path).parent
            download = input(
                "Do you want to download and extract the file? Y/N ")
            if download == "Y":
                download_url = "https://archive.org/download/mario-kart-64-usa/Mario%20Kart%2064%20%28USA%29.zip"
                os.system(f"wget {download_url} -O /tmp/marioKart.zip")
                os.system(
                    f"unzip /tmp/marioKart.zip -d {str(rom_dir.resolve())}")
                os.system(
                    f"mv '{str(rom_dir.resolve() / 'Mario Kart 64 (USA).n64')}' {rom_path}")
                cprint("Rom file downloaded!")
            else:
                raise Exception(msg)

    def _start_emulator(self,
                        rom_name,
                        gfx_plugin,
                        input_driver_path,
                        res_w=SCR_W,
                        res_h=SCR_H,
                        res_d=SCR_D):

        rom_path = os.path.abspath(
            os.path.join(os.path.dirname(inspect.stack()[0][1]),
                         '../ROMs',
                         rom_name))
        self.check_rom_path(rom_path)


        input_driver_path = os.path.abspath(
            os.path.expanduser(input_driver_path))
        if not os.path.isfile(input_driver_path):
            msg = "Input driver not found: " + input_driver_path
            cprint(msg, 'red')
            raise Exception(msg)

        benchmark_options = [
            "--nospeedlimit",
            "--set", f"Core[CountPerOp]={COUNTPEROP}",
            # "--set", f"Core[DelaySI]=False",
            # "--set", f"Video-Glide64[filtering]=0",
            # "--set", f"Video-Glide64[fast_crc]=True",
            # "--set", f"Video-Glide64[fb_hires]=False",
            # "--set", "Video-Rice[ScreenUpdateSetting]=3",
            # "--set", "Video-Rice[FastTextureLoading]=1",
            # "--set", "Video-Rice[TextureQuality]=2",
            # "--set", "Video-Rice[ColorQuality]=1",
        ]

        cmd = [self.config['MUPEN_CMD'],
               # "--nospeedlimit",
               "--nosaveoptions",
               "--resolution",
               "%ix%i" % (res_w, res_h),
               "--gfx", gfx_plugin,
               "--audio", "dummy",
               "--set", f"Input-Bot-Control0[port]={self.input_port}",
               "--input", input_driver_path,
               rom_path]

        if self.benchmark:
            cmd = [cmd[0]] + benchmark_options + cmd[1:]

        xvfb_proc = None
        if self.config['USE_XVFB']:
            display_num = 0
            success = False
            # If we couldn't find an open display number after 15 attempts, give up
            while not success and display_num <= 99:
                display_num += 1
                xvfb_cmd = [self.config['XVFB_CMD'],
                            ":" + str(display_num),
                            "-screen",
                            "0",
                            "%ix%ix%i" % (res_w, res_h, res_d * 8),
                            "-noreset",
                            "-fbdir",
                            self.config['TMP_DIR']]

                cprint('Starting xvfb with command: %s' % xvfb_cmd, 'yellow')

                xvfb_proc = subprocess.Popen(
                    xvfb_cmd, shell=False, stderr=subprocess.STDOUT)

                time.sleep(2)  # Give xvfb a couple seconds to start up

                # Poll the process to see if it exited early
                # (most likely due to a server already active on the display_num)
                if xvfb_proc.poll() is None:
                    success = True

                print('')  # new line

            if not success:
                msg = "Failed to initialize Xvfb!"
                cprint(msg, 'red')
                raise Exception(msg)

            os.environ["DISPLAY"] = ":" + str(display_num)
            cprint('Using DISPLAY %s' % os.environ["DISPLAY"], 'blue')
            cprint('Changed to DISPLAY %s' % os.environ["DISPLAY"], 'red')
            
            os.environ["XVFB_FB_PATH"] = self.config["TMP_DIR"] + "/Xvfb_screen0"
            cmd = [self.config['VGLRUN_CMD'],
                   "-d", ":" + str(display_num)] + cmd
        # else:
        #     cmd.append("--noosd")

        cprint('Starting emulator with comand: %s' % cmd, 'yellow')

        print("COMMAND: ", " ".join(cmd))

        emulator_process = subprocess.Popen(cmd,
                                            env=os.environ.copy(),
                                            shell=False,
                                            stderr=subprocess.STDOUT)

        emu_mon = EmulatorMonitor()
        monitor_thread = threading.Thread(target=emu_mon.monitor_emulator,
                                          args=[emulator_process])
        monitor_thread.daemon = True
        monitor_thread.start()

        return xvfb_proc, emulator_process

    def _kill_emulator(self):
        cprint('Kill Emulator called!', 'yellow')
        try:
            self._act(ControllerState.NO_OP)
            if self.emulator_process is not None:
                self.emulator_process.kill()
            if self.xvfb_process is not None:
                self.xvfb_process.terminate()
            if self.containerized:
                subprocess.run(["docker", "kill", self.container_name])
        except AttributeError:
            pass  # We may be shut down during intialization before these attributes have been set

    def reset(self):
        return self._reset()

    def step(self, action):
        return self._step(action)

    def render(self, mode):
        return self._render(mode)


###############################################
class EmulatorMonitor:
    def monitor_emulator(self, emulator):
        emu_return = emulator.poll()
        while emu_return is None:
            time.sleep(2)
            if emulator is not None:
                emu_return = emulator.poll()
            else:
                print('Emulator reference is no longer valid. Shutting down?')
                return

        # TODO: this means our environment died... need to die too
        print('Emulator closed with code: ' + str(emu_return))


###############################################
class ControllerState(object):

    # Controls           [ JX,  JY,  A,  B, RB, LB,  Z, CR, CL, CD, CU, DR, DL, DD, DU,  S]
    NO_OP = [0,   0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]
    START_BUTTON = [0,   0,  0,  0,  0,  0,
                    0,  0,  0,  0,  0,  0,  0,  0,  0,  1]
    A_BUTTON = [0,   0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]
    B_BUTTON = [0,   0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]
    RB_BUTTON = [0,   0,  0,  0,  1,  0,  0,
                 0,  0,  0,  0,  0,  0,  0,  0,  0]
    CR_BUTTON = [0,   0,  0,  0,  0,  0,  0,
                 1,  0,  0,  0,  0,  0,  0,  0,  0]
    CL_BUTTON = [0,   0,  0,  0,  0,  0,  0,
                 0,  1,  0,  0,  0,  0,  0,  0,  0]
    CD_BUTTON = [0,   0,  0,  0,  0,  0,  0,
                 0,  0,  1,  0,  0,  0,  0,  0,  0]
    CU_BUTTON = [0,   0,  0,  0,  0,  0,  0,
                 0,  0,  0,  1,  0,  0,  0,  0,  0]
    JOYSTICK_UP = [0,  127, 0,  0,  0,  0,
                   0,  0,  0,  0,  0,  0,  0,  0,  0,  0]
    JOYSTICK_DOWN = [0, -128, 0,  0,  0,  0,
                     0,  0,  0,  0,  0,  0,  0,  0,  0,  0]
    JOYSTICK_LEFT = [-128,  0,  0,  0,  0,  0,
                     0,  0,  0,  0,  0,  0,  0,  0,  0,  0]
    JOYSTICK_RIGHT = [127,  0,  0,  0,  0,  0,
                      0,  0,  0,  0,  0,  0,  0,  0,  0,  0]

    def __init__(self, controls=NO_OP):
        # print("doing controls:", controls)
        self.X_AXIS = controls[0]
        self.Y_AXIS = controls[1]
        self.A_BUTTON = controls[2]
        self.B_BUTTON = controls[3]
        self.R_TRIG = controls[4]
        self.L_TRIG = controls[5]
        self.Z_TRIG = controls[6]
        self.R_CBUTTON = controls[7]
        self.L_CBUTTON = controls[8]
        self.D_CBUTTON = controls[9]
        self.U_CBUTTON = controls[10]
        self.R_DPAD = controls[11]
        self.L_DPAD = controls[12]
        self.D_DPAD = controls[13]
        self.U_DPAD = controls[14]
        self.START_BUTTON = controls[15]
        self.controls = controls

    def to_msg(self):
        return "|".join([str(i) for i in self.controls])

###############################################


class ControllerUpdater(object):
    BUFFER_SIZE = 4096

    def __init__(self, input_host, input_port, control_timeout, frame_skip, num_pixels):
        self.control_timeout = control_timeout
        self.controls = ControllerState()
        self.input_host = input_host
        self.image_buffer_size = num_pixels * 3
        self.input_port = input_port
        self.running = True
        self.frame_skip = frame_skip
        self.last_image = None
        self.frame_skip_enabled = True

    def send_controls(self, controls, count=None, force_count=False):
        if not self.running:
            return
        self.controls = controls
        msg = self.controls.to_msg()
        frame_skip = count if count is not None else self.frame_skip
        msg += f"|{frame_skip if self.frame_skip_enabled or force_count else 0}#"
        msg = "#|" + (msg * 3)
        image = "none".encode()
        while (image == "none".encode() or len(image) < 10):
            try:
                self.socket.sendall(msg.encode())
                image = b''
                while True:
                    content = self.socket.recv(self.BUFFER_SIZE)
                    # print("got content", len(content))
                    if not content:
                        break
                    image += content
                    if len(content) < self.BUFFER_SIZE:
                        break
            except:
                # reconnect
                while True:
                    try:
                        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        self.socket.connect((self.input_host, int(self.input_port)))
                        self.socket.sendall(msg.encode())
                        image = b''
                        while True:
                            content = self.socket.recv(self.BUFFER_SIZE)
                            if not content:
                                break
                            image += content
                            if len(content) < self.BUFFER_SIZE:
                                break
                        break
                    except Exception as e:
                        cprint(f"cannot connect: {e}, retrying...")
                
        if len(image) > self.image_buffer_size:
            image = image[:self.image_buffer_size]
        if self.image_buffer_size != len(image):
            return self.last_image
        self.last_image = image
        return image
        

    def shutdown(self):
        self.running = False
        self.socket.close()

    @contextmanager
    def frame_skip_disabled(self):
        self.frame_skip_enabled = False
        yield True
        self.frame_skip_enabled = True


###############################################
