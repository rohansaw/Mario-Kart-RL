from PIL import Image
from pathlib import Path
import sys
import socket

PY3_OR_LATER = sys.version_info[0] >= 3


import abc
import wandb
import array
from contextlib import contextmanager
import inspect
import itertools
import json
import os
import subprocess
import threading
import time
from termcolor import cprint
import yaml

import gym
from gym import error, spaces, utils
from gym.utils import seeding

import numpy as np

import mss

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

    def __init__(self, benchmark=True, resolution="supersmall", res_w=None, res_h=None, variable_episode_length=False, base_episode_length=20000, episode_length_increase=1, gray_scale=True):
        
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
        
        self.viewer = None
        self.benchmark = benchmark
        self.reset_count = 0
        self.step_count = 0
        self.running = True
        self.episode_aborted = False
        self.episode_completed = False
        self.episode_reward = 0
        self.last_episode_reward = 0
        self.max_duration = 0
        self.max_reward = 0
        self.pixel_array = None
        self.gray_scale = gray_scale
        self._base_load_config()
        self._base_validate_config()
        self.frame_skip = self.config['FRAME_SKIP']
        if self.frame_skip < 1:
            self.frame_skip = 1
        
        
        self.config["PORT_NUMBER"] = self._next_free_port(self.config["PORT_NUMBER"])
        self.controller_server = self._start_controller_server()


        initial_disp = os.environ["DISPLAY"]
        cprint('Initially on DISPLAY %s' % initial_disp, 'red')

        # If the EXTERNAL_EMULATOR environment variable is True, we are running the
        # emulator out-of-process (likely via docker/docker-compose). If not, we need
        # to start the emulator in-process here
        external_emulator = "EXTERNAL_EMULATOR" in os.environ and os.environ["EXTERNAL_EMULATOR"] == 'True'
        if not external_emulator:
            self.xvfb_process, self.emulator_process = \
                self._start_emulator(rom_name=self.config['ROM_NAME'],
                                     gfx_plugin=self.config['GFX_PLUGIN'],
                                     input_driver_path=self.config['INPUT_DRIVER_PATH'],
                                     res_w=SCR_W, res_h=SCR_H, res_d=SCR_D)

        # TODO: Test and cleanup:
        # May need to initialize this after the DISPLAY env var has been set
        # so it attaches to the correct X display; otherwise screenshots may
        # come from the wrong place. This used to be true when we were using
        # wxPython for screenshots. Untested after switching to mss.
        cprint('Calling mss.mss() with DISPLAY %s' % os.environ["DISPLAY"], 'red')
        self.mss_grabber = mss.mss()
        time.sleep(2) # Give mss a couple seconds to initialize; also may not be necessary

        # Restore the DISPLAY env var
        os.environ["DISPLAY"] = initial_disp
        cprint('Changed back to DISPLAY %s' % os.environ["DISPLAY"], 'red')
        

        with self.controller_server.frame_skip_disabled():
            self._navigate_menu()

        self.observation_space = \
            spaces.Box(low=0, high=255, shape=(SCR_H, SCR_W, 1), dtype=np.uint8)

        actions = [[-80, 80], # Joystick X-axis
                                                  [-80, 80], # Joystick Y-axis
                                                  [  0,  1], # A Button
                                                  [  0,  1], # B Button
                                                  [  0,  1], # RB Button
                                                  [  0,  1], # LB Button
                                                  [  0,  1], # Z Button
                                                  [  0,  1], # C Right Button
                                                  [  0,  1], # C Left Button
                                                  [  0,  1], # C Down Button
                                                  [  0,  1], # C Up Button
                                                  [  0,  1], # D-Pad Right Button
                                                  [  0,  1], # D-Pad Left Button
                                                  [  0,  1], # D-Pad Down Button
                                                  [  0,  1], # D-Pad Up Button
                                                  [  0,  1], # Start Button
                                                 ]

        self.action_space = spaces.MultiDiscrete([len(action) for action in actions])

    def _base_load_config(self):
        self.config = yaml.safe_load(open(os.path.join(os.path.dirname(inspect.stack()[0][1]), "config.yml")))
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
        #cprint('Step %i: %s' % (self.step_count, action), 'green')
        # start = time.time()
        self._act(action)
        # # end = time.time()
        # print("step time:", end - start)
        # # start = time.time()
        obs = self._observe()
        # # end = time.time()
        # print("observe time:", end - start)
        # # start = time.time()
        if self.step_count >= self.episode_length:
            cprint("aborting episode due to max steps reached!", "cyan")
            self.episode_aborted = True
            self.episode_completed = False
        else:
            self.episode_completed, self.episode_aborted = self._evaluate_end_state()
        # # end = time.time()
        # print("_evaluate_end_state time:", end - start)
        # # start = time.time()
        reward = self._get_reward()
        # # end = time.time()
        # print("_get_reward time:", end - start)

        self.step_count += 1
        # if self.episode_over:
        self.episode_reward += reward
        
        if self.gray_scale:
            obs = np.average(obs, axis=2, weights=[0.299, 0.587, 0.114], keepdims=True).astype(np.uint8)
            # self.pixel_array = np.dot(self.pixel_array[...,:3], [0.299, 0.587, 0.114])
        if self.episode_aborted:
            cprint("Episode aborted!", "cyan")
        if self.episode_completed:
            cprint("Episode successfully completed!", "cyan")
            if wandb.run is not None:
                wandb.log({"env/episode-stop-reason": 3})
        return obs, reward, self.episode_aborted or self.episode_completed, {}

    def _act(self, action, count=1, force_count=False):
        # print("got action:", action, "count:", count)
        if not self.controller_server.frame_skip_enabled and not force_count:
            # print("sending single passes")
            for _ in itertools.repeat(None, count):
                self.controller_server.send_controls(ControllerState(action))
            # print("sending...")
        else:
            self.controller_server.send_controls(ControllerState(action), count=count, force_count=force_count)
        # self.render(mode="human")
        # time.sleep(0.2)
        # print("done.")

    def _wait(self, count=1, wait_for='Unknown'):
        self._act(ControllerState.NO_OP, count=count, force_count=True)

    def _press_button(self, button, times=1):
        for _ in itertools.repeat(None, times):
            self._act(button) # Press
            self._act(ControllerState.NO_OP) # and release
    
    def _observe(self):
        #cprint('Observe called!', 'yellow')

        if self.config['USE_XVFB']:
            offset_x = 0
            offset_y = 0
        else:
            offset_x = self.config['OFFSET_X']
            offset_y = self.config['OFFSET_Y']
        image_array = \
            np.array(self.mss_grabber.grab({"top": offset_y,
                                            "left": offset_x,
                                            "width": SCR_W,
                                            "height": SCR_H}),
                    dtype=np.uint8)
    
        # drop the alpha channel and flip red and blue channels (BGRA -> RGB)
        self.pixel_array = np.flip(image_array[:, :, :3], 2)
        return self.pixel_array

    @abc.abstractmethod
    def _navigate_menu(self):
        return

    @abc.abstractmethod
    def _get_reward(self):
        #cprint('Get Reward called!', 'yellow')
        return 0

    @abc.abstractmethod
    def _evaluate_end_state(self):
        #cprint('Evaluate End State called!', 'yellow')
        return False, False

    @abc.abstractmethod
    def _reset(self):
        cprint('Reset called!', 'yellow')
        self.reset_count += 1
        self.last_episode_reward = self.episode_reward
        
        self.max_reward = max(self.max_reward, self.episode_reward)
        self.max_duration = max(self.max_duration, self.step_count)
        cprint(f"last episode reward: {self.episode_reward:.1f}, duration: {self.step_count}", "green")
        
        if wandb.run is not None:
            wandb.log({
                "env/rewards": self.episode_reward,
                "env/length": self.step_count,
                "env/max-reward": self.max_reward,
                "env/max-duration": self.max_duration,
            })
        self.episode_reward = 0
        if self.reset_count > 1 and self.variable_episode_length:
            self.episode_length += self.episode_length_increase
            cprint(f"next episode length: {self.episode_length}", "yellow")
        

        self.step_count = 0
        obs = self._observe()
        if self.gray_scale:
            obs = np.average(obs, axis=2, weights=[0.299, 0.587, 0.114], keepdims=True).astype(np.uint8)
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
        server = ControllerUpdater(
            input_host  = '',
            input_port= self.config['PORT_NUMBER'],
            control_timeout = self.config['ACTION_TIMEOUT'],
            frame_skip = self.frame_skip) # TODO: Environment argument (with issue #26)
        print('ControllerUpdater started on port ', self.config['PORT_NUMBER'])
        return server

    def _stop_controller_server(self):
        #cprint('Stop Controller Server called!', 'yellow')
        if hasattr(self, 'controller_server'):
            self.controller_server.shutdown()

    def _next_free_port(self, port):
        max_ports_to_test = 30
        for i in range(port, port + max_ports_to_test):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    print("trying out port", i, "...")
                    s.bind(('localhost', i))
                    return i
                except:
                    pass
        raise Exception(f"cannot find any available port in range {port} - {port + max_ports_to_test}")

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

        if not os.path.isfile(rom_path):
            msg = "ROM not found: " + rom_path
            cprint(msg, 'red')
            rom_dir = Path(rom_path).parent
            download = input("Do you want to download and extract the file? Y/N ")
            if download == "Y":
                download_url = "https://archive.org/download/mario-kart-64-usa/Mario%20Kart%2064%20%28USA%29.zip"
                os.system(f"wget {download_url} -O /tmp/marioKart.zip")
                os.system(f"unzip /tmp/marioKart.zip -d {str(rom_dir.resolve())}")
                os.system(f"mv '{str(rom_dir.resolve() / 'Mario Kart 64 (USA).n64')}' {rom_path}")
                cprint("Rom file downloaded!")
            else:
                raise Exception(msg)
                

        input_driver_path = os.path.abspath(os.path.expanduser(input_driver_path))
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
                "--set", f"Input-Bot-Control0[port]={self.config['PORT_NUMBER']}",
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

                xvfb_proc = subprocess.Popen(xvfb_cmd, shell=False, stderr=subprocess.STDOUT)

                time.sleep(2) # Give xvfb a couple seconds to start up

                # Poll the process to see if it exited early
                # (most likely due to a server already active on the display_num)
                if xvfb_proc.poll() is None:
                    success = True

                print('') # new line

            if not success:
                msg = "Failed to initialize Xvfb!"
                cprint(msg, 'red')
                raise Exception(msg)

            os.environ["DISPLAY"] = ":" + str(display_num)
            cprint('Using DISPLAY %s' % os.environ["DISPLAY"], 'blue')
            cprint('Changed to DISPLAY %s' % os.environ["DISPLAY"], 'red')

            cmd = [self.config['VGLRUN_CMD'], "-d", ":" + str(display_num)] + cmd
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
        #cprint('Kill Emulator called!', 'yellow')
        try:
            self._act(ControllerState.NO_OP)
            if self.emulator_process is not None:
                self.emulator_process.kill()
            if self.xvfb_process is not None:
                self.xvfb_process.terminate()
        except AttributeError:
            pass # We may be shut down during intialization before these attributes have been set

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
    NO_OP              = [  0,   0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]
    START_BUTTON       = [  0,   0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1]
    A_BUTTON           = [  0,   0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]
    B_BUTTON           = [  0,   0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]
    RB_BUTTON          = [  0,   0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]
    CR_BUTTON          = [  0,   0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0]
    CL_BUTTON          = [  0,   0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0]
    CD_BUTTON          = [  0,   0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0]
    CU_BUTTON          = [  0,   0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0]
    JOYSTICK_UP        = [  0,  127, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]
    JOYSTICK_DOWN      = [  0, -128, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]
    JOYSTICK_LEFT      = [-128,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]
    JOYSTICK_RIGHT     = [ 127,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]

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

    def __init__(self, input_host, input_port, control_timeout, frame_skip):
        self.control_timeout = control_timeout
        self.controls = ControllerState()
        self.input_host = input_host
        self.input_port = input_port
        self.running = True
        self.frame_skip = frame_skip
        self.frame_skip_enabled = True

    def send_controls(self, controls, count=None, force_count=False):
        if not self.running:
            return
        self.controls = controls
        msg = self.controls.to_msg()
        frame_skip = count if count is not None else self.frame_skip
        msg += f"|{frame_skip if self.frame_skip_enabled or force_count else 0}#"
        
        try:
            self.socket.sendall(msg.encode())
            self.socket.recv(1)
        except:
            # reconnect
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.input_host, self.input_port))
            self.socket.sendall(msg.encode())
            self.socket.recv(1)
    def shutdown(self):
        self.running = False
        self.socket.close()

    @contextmanager
    def frame_skip_disabled(self):
        self.frame_skip_enabled = False
        yield True
        self.frame_skip_enabled = True


###############################################
