from pathlib import Path
import sys
import socket

PY3_OR_LATER = sys.version_info[0] >= 3


import abc
import inspect
import os
import subprocess
import threading
import time
from termcolor import cprint
import yaml


import mss

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

BENCHMARK = False


###############################################
class EnvController():
    metadata = {'render.modes': ['human', 'rgb_array']}
    
    resolutions = {
        "normal": (640, 480),
        "small": (320, 240),
        # the logical next resolution would have been 160, 120. however the progress bar is not rendered properly in that resolution, so a slightly larger ones is picked instead
        "supersmall": (170, 128),
    }

    def __init__(self, benchmark=True, resolution="supersmall", res_w=None, res_h=None, auto_abort=True, variable_episode_length=False, base_episode_length=20000, episode_length_increase=1, gray_scale=True):
        
        global SCR_W, SCR_H
        if res_w is not None and res_h is not None:
            self.res_w = res_w
            self.res_h = res_h
        else:
            self.res_w, self.res_h = self.resolutions[resolution]
        SCR_W, SCR_H = self.res_w, self.res_h
        
        self.benchmark = benchmark
        self.running = True
        self._base_load_config()
        self._base_validate_config()
        self.frame_skip = self.config['FRAME_SKIP']
        if self.frame_skip < 1:
            self.frame_skip = 1
        
        
        self.config["PORT_NUMBER"] = self._next_free_port(self.config["PORT_NUMBER"])


        initial_disp = os.environ["DISPLAY"]
        cprint('Initially on DISPLAY %s' % initial_disp, 'red')

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

    def _next_free_port(self, port):
        max_ports_to_test = 30
        for i in range(port, port + max_ports_to_test):
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                print("trying out port", i, "...")
                s.bind(('localhost', i))
                return i
            except:
                pass
            finally:
                s.close()
        raise Exception("cannot find any available port in range" +  port + "port + max_ports_to_test")

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
            '''download = input("Do you want to download and extract the file? Y/N ")
            if download == "Y":
                download_url = "https://archive.org/download/mario-kart-64-usa/Mario%20Kart%2064%20%28USA%29.zip"
                os.system("wget " + {download_url} -O /tmp/marioKart.zip")
                os.system(f"unzip /tmp/marioKart.zip -d {str(rom_dir.resolve())}")
                os.system(f"mv '{str(rom_dir.resolve() / 'Mario Kart 64 (USA).n64')}' {rom_path}")
                cprint("Rom file downloaded!")
            else: '''
            raise Exception(msg)
                

        input_driver_path = os.path.abspath(os.path.expanduser(input_driver_path))
        if not os.path.isfile(input_driver_path):
            msg = "Input driver not found: " + input_driver_path
            cprint(msg, 'red')
            raise Exception(msg)
        
        benchmark_options = [
                "--nospeedlimit",
                "--set", "Core[CountPerOp]="+str(COUNTPEROP),
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
                "--set", "Input-Bot-Control0[port]="+str(self.config['PORT_NUMBER']),
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
        try:
            if self.emulator_process is not None:
                self.emulator_process.kill()
            if self.xvfb_process is not None:
                self.xvfb_process.terminate()
        except AttributeError:
            pass # We may be shut down during intialization before these attributes have been set

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

if __name__ == '__main__':
    controller = EnvController()