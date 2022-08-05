import subprocess
import os
configs = [{'input_port': '8082', 'vnc_port': '8083',
            'xvfb_port': '6000', 'screen_port': '9000', 'name': '1'}]
os.environ["EXTERNAL_EMULATOR"] = 'True'
for config in configs:
    os.environ["INPUT_PORT"] = config['input_port']
    os.environ["VNC_PORT"] = config['vnc_port']
    os.environ["XVFB_PORT"] = config['xvfb_port']
    os.environ["SCREEN_PORT"] = config['screen_port']
    subprocess.run(
        f"docker-compose --project-name {config['name']} up --build -d", shell=True)
