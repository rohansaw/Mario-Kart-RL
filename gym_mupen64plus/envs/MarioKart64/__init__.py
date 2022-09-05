from gym.envs.registration import register

from gym_mupen64plus.envs.MarioKart64.mario_kart_env import MarioKartEnv
from gym_mupen64plus.envs.MarioKart64.discrete_envs import MarioKartDiscreteEnv
import subprocess
import os
courses = [
    {'name': 'Luigi-Raceway',      'cup': 'Mushroom', 'max_steps': 1250},
    {'name': 'Moo-Moo-Farm',       'cup': 'Mushroom', 'max_steps': 1250},
    {'name': 'Koopa-Troopa-Beach', 'cup': 'Mushroom', 'max_steps': 1250},
    {'name': 'Kalimari-Desert',    'cup': 'Mushroom', 'max_steps': 1250},
    {'name': 'Toads-Turnpike',     'cup': 'Flower',   'max_steps': 1250},
    {'name': 'Frappe-Snowland',    'cup': 'Flower',   'max_steps': 1250},
    {'name': 'Choco-Mountain',     'cup': 'Flower',   'max_steps': 1250},
    {'name': 'Mario-Raceway',      'cup': 'Flower',   'max_steps': 1250},
    {'name': 'Wario-Stadium',      'cup': 'Star',     'max_steps': 1250},
    {'name': 'Sherbet-Land',       'cup': 'Star',     'max_steps': 1250},
    {'name': 'Royal-Raceway',      'cup': 'Star',     'max_steps': 1250},
    {'name': 'Bowsers-Castle',     'cup': 'Star',     'max_steps': 1250},
    {'name': 'DKs-Jungle-Parkway', 'cup': 'Special',  'max_steps': 1250},
    {'name': 'Yoshi-Valley',       'cup': 'Special',  'max_steps': 1250},
    {'name': 'Banshee-Boardwalk',  'cup': 'Special',  'max_steps': 1250},
    {'name': 'Rainbow-Road',       'cup': 'Special',  'max_steps': 1250},
]


configs = [{'input_port': '8082', 'vnc_port': '8083', 'xvfb_port': '6000', 'screen_port': '9000', 'name': '1'},
           {'input_port': '8084', 'vnc_port': '8085', 'xvfb_port': '6001', 'screen_port': '9001', 'name': '2'}]
os.environ["EXTERNAL_EMULATOR"] = 'True'
# here docker container starten und ports setzen und dann leggo
# for config in configs:
for course in courses:
    # os.environ["INPUT_PORT"] = config['input_port']
    # os.environ["VNC_PORT"] = config['vnc_port']
    # os.environ["XVFB_PORT"] = config['xvfb_port']
    # os.environ["SCREEN_PORT"] = config['screen_port']
    # subprocess.run(
    #     f"docker-compose --project-name {config['name']} up --build -d", shell=True)
    # Continuous Action Space:
    register(
        id='Mario-Kart-%s-v0' % course['name'],
        entry_point='gym_mupen64plus.envs.MarioKart64:MarioKartEnv',
        kwargs={'course': course['name'].replace('-', '')},
        # tags={
        #    'mupen': True,
        #    'cup': course['cup'],
        # #    'wrapper_config.TimeLimit.max_episode_steps': course['max_steps'],

        # },
        # max_episode_steps=course["max_steps"],
        nondeterministic=True,
    )

    # Discrete Action Space:
    register(
        id='Mario-Kart-Discrete-%s-v0' % course['name'],
        entry_point='gym_mupen64plus.envs.MarioKart64:MarioKartDiscreteEnv',
        kwargs={'course': course['name'].replace('-', '')},
        # kwargs={'course': course['name'].replace(
        #     '-', ''), 'input_port': config['input_port'], 'vnc_port': config['vnc_port']},
        # tags={
        #    'mupen': True,
        #    'cup': course['cup'],
        #    'wrapper_config.TimeLimit.max_episode_steps': course['max_steps'],
        # },
        # max_episode_steps=course["max_steps"],
        nondeterministic=True,
    )
