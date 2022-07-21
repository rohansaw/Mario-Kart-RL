from PIL import Image
import time
import abc
import inspect
import random
import itertools
import os
import yaml
from termcolor import cprint
from gym import spaces
from gym_mupen64plus.envs.mupen64plus_env \
  import Mupen64PlusEnv, ControllerState, IMAGE_HELPER
import numpy as np

###############################################
class MarioKartEnv(Mupen64PlusEnv):
    __metaclass__ = abc.ABCMeta

    # Indicates the color value of the pixel at point (203, 51)
    # This is where the lap number is present in the default HUD
    END_RACE_PIXEL_COLORS = {"mupen64plus-video-rice.so"       : ( 66,  49,  66),
                             "mupen64plus-video-glide64mk2.so" : (214, 148, 214),
                             "mupen64plus-video-glide64.so"    : (157, 112, 158)}

    HUD_PROGRESS_COLOR_VALUES = {(000, 000, 255): 0, #   Blue: Lap 1
                                 (255, 255, 000): 1, # Yellow: Lap 2
                                 (255, 000, 000): 2} #    Red: Lap 3

    CHARACTERS = {
        'mario'  : (0, 0),
        'luigi'  : (0, 1),
        'peach'  : (0, 2),
        'toad'   : (0, 3),
        'yoshi'  : (1, 0),
        'd.k.'   : (1, 1),
        'wario'  : (1, 2),
        'bowser' : (1, 3)
    }
    
    COURSES = {
        'LuigiRaceway'     : (0, 0),
        'MooMooFarm'       : (0, 1),
        'KoopaTroopaBeach' : (0, 2),
        'KalimariDesert'   : (0, 3),
        'ToadsTurnpike'    : (1, 0),
        'FrappeSnowland'   : (1, 1),
        'ChocoMountain'    : (1, 2),
        'MarioRaceway'     : (1, 3),
        'WarioStadium'     : (2, 0),
        'SherbetLand'      : (2, 1),
        'RoyalRaceway'     : (2, 2),
        'BowsersCastle'    : (2, 3),
        'DKsJungleParkway' : (3, 0),
        'YoshiValley'      : (3, 1),
        'BansheeBoardwalk' : (3, 2),
        'RainbowRoad'      : (3, 3)
    }

    DEFAULT_STEP_REWARD = -0.1
    LAP_REWARD = 200
    CHECKPOINT_REWARD = 0.5
    BACKWARDS_PUNISHMENT = 3
    END_REWARD = 1000
    
    PROGRESS_SCALE = 1
    PROGRESS_REWARD = 1.0

    END_EPISODE_THRESHOLD = 0

    PLAYER_ROW = 0
    PLAYER_COL = 0

    MAP_SERIES = 0
    MAP_CHOICE = 0

    ENABLE_CHECKPOINTS = False

    AMOUNT_STEPS_CONSIDERED_STUCK = 40
    MIN_PROGRESS = 1.5
    
    CHECKPOINTS = {
        160: [16, 9, 146, 111],
        170: [17, 10, 155, 118],
        320: [32, 18, 292, 222],
        640: [64, 36, 584, 444],
    }

    END_PIXELS = {
        160: [50, 12],
        170: [60, 18],
        320: [101, 25],
        640: [203, 51],
    }

    def __init__(self, character='mario', course='LuigiRaceway', random_tracks=False, **kwargs):
        self._set_character(character)
        self._set_course(course)
        super(MarioKartEnv, self).__init__(**kwargs)

        self.end_race_pixel_color = self.END_RACE_PIXEL_COLORS[self.config["GFX_PLUGIN"]]

        actions = [[-80, 80],  # Joystick X-axis
                    [-80, 80],  # Joystick Y-axis
                    [  0,  1],  # A Button
                    [  0,  1],  # B Button
                    [  0,  1]]  # RB Button
        
        self.action_space = spaces.MultiDiscrete([len(action) for action in actions])
        
        self.random_tracks = random_tracks
        self.checkpoints = self.CHECKPOINTS[self.res_w]
        self.CHECKPOINT_LOCATIONS = list(self._generate_checkpoints(*self.checkpoints))

    def _load_config(self):
        self.config.update(yaml.safe_load(open(os.path.join(os.path.dirname(inspect.stack()[0][1]), "mario_kart_config.yml"))))
        
    def _validate_config(self):
        # print("validate sub")
        gfx_plugin = self.config["GFX_PLUGIN"]
        if gfx_plugin not in self.END_RACE_PIXEL_COLORS:
            raise AssertionError("Video Plugin '" + gfx_plugin + "' not currently supported by MarioKart environment")

    def _step(self, action):
        # Interpret the action choice and get the actual controller state for this step
        controls = action + [  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]
        return super(MarioKartEnv, self)._step(controls)

    def _reset_after_race(self):
        print("resetting after race")
        self._wait(count=275, wait_for='times screen')
        self._navigate_post_race_menu()
        self._wait(count=40, wait_for='map select screen')
        self._navigate_map_select()
        self._wait(count=50, wait_for='race to load')

    def _reset_during_race(self):
        print("resetting during race")
        # Can't pause the race until the light turns green
        if (self.step_count * self.controller_server.frame_skip) < 120:
            steps_to_wait = 100 - (self.step_count * self.controller_server.frame_skip)
            self._wait(count=steps_to_wait, wait_for='green light so we can pause')
        self._press_button(ControllerState.START_BUTTON)
        self._press_button(ControllerState.JOYSTICK_DOWN)
        self._press_button(ControllerState.A_BUTTON)
        self._wait(count=76, wait_for='race to load')
    
    def _reset_during_race_change_course(self):
        print("resetting during race CHANGING COURSE")
        # Can't pause the race until the light turns green
        if (self.step_count * self.controller_server.frame_skip) < 120:
            steps_to_wait = 100 - (self.step_count * self.controller_server.frame_skip)
            self._wait(count=steps_to_wait, wait_for='green light so we can pause')
        self._press_button(ControllerState.START_BUTTON)
        self._press_button(ControllerState.JOYSTICK_DOWN)
        self._press_button(ControllerState.JOYSTICK_DOWN)
        self._press_button(ControllerState.A_BUTTON)
        self._wait(count=31, wait_for='race to load')
        
        self._navigate_map_select()
        
        self._wait(count=46, wait_for='race to load')

    def _reset(self):
        self.lap = 0
        self.step_count_at_lap = 0
        self._last_progress_point = 0
        self.last_known_lap = -1
        self._last_progresses = []


        # Nothing to do on the first call to reset()
        if self.reset_count > 0:
            # Make sure we don't skip frames while navigating the menus
            with self.controller_server.frame_skip_disabled():
                if self.random_tracks:
                    self._set_course(random.choice(list(self.COURSES.keys())))
                    self._reset_during_race_change_course()
                elif self.episode_complete:
                    self._reset_after_race()
                else:
                    self._reset_during_race()

        self.episode_aborted = False
        self.episode_complete = False
        return super(MarioKartEnv, self)._reset()

    def reset(self):
        return self._reset()

    def _save_last_progress_point(self, idx):
        if len(self._last_progresses) >=self.AMOUNT_STEPS_CONSIDERED_STUCK: 
            self._last_progresses.pop(0)
        self._last_progresses.append(idx)

    def _get_progress(self):
        idx = self._last_progress_point
        value_of_last_progress_point = self._evaluate_checkpoint([self.CHECKPOINT_LOCATIONS[idx]])
        if idx == 0 and value_of_last_progress_point == -1:
            return 0
        
        # should be the case if we went backwards
        if value_of_last_progress_point != self.lap:
            while(True):
                idx = ((idx - 1) + len(self.CHECKPOINT_LOCATIONS)) % len(self.CHECKPOINT_LOCATIONS)
                if idx == self._last_progress_point:
                    print("went one time around!", self._last_progress_point, self._evaluate_checkpoint([self.CHECKPOINT_LOCATIONS[self._last_progress_point]]), self.lap)
                    self._last_progress_point = 0
                    return -1.0
                if self._evaluate_checkpoint([self.CHECKPOINT_LOCATIONS[idx]]) == self.lap:
                    break
        else:
            while(True):
                idx = (idx + 1) % len(self.CHECKPOINT_LOCATIONS)
                if idx == self._last_progress_point:
                    print("went one time around!", self._last_progress_point, self._evaluate_checkpoint([self.CHECKPOINT_LOCATIONS[self._last_progress_point]]), self.lap)
                    self._last_progress_point = 0
                    return - 1.0
                if self._evaluate_checkpoint([self.CHECKPOINT_LOCATIONS[idx]]) != self.lap:
                    break
        idx = ((idx - 1) + len(self.CHECKPOINT_LOCATIONS)) % len(self.CHECKPOINT_LOCATIONS)
        dist = idx - self._last_progress_point
        # if we got into a new lap, we have to get the real progress
        if abs(dist) > (len(self.CHECKPOINT_LOCATIONS) // 2):
            dist = len(self.CHECKPOINT_LOCATIONS) - abs(dist)
            self._last_progresses = []
        self._last_progress_point = idx
        self._save_last_progress_point(idx)
        return dist * self.PROGRESS_SCALE

    def _get_reward(self):
        #cprint('Get Reward called!','yellow')

        reward_to_return = 0
        cur_lap = self._get_lap()

        if self.episode_completed:
            cprint("yayy, race completed!!")
            # Scale out the end reward based on the total steps to get here; the fewer steps, the higher the reward
            reward_to_return = 5 * (1250 - self.step_count) + self.END_REWARD #self.END_REWARD * (5000 / self.step_count) - 3000
        else:
            if cur_lap > self.lap:
                self.lap = cur_lap
                cprint('Lap %s!' % self.lap, 'green')

                # Scale out the lap reward based on the steps to get here; the fewer steps, the higher the reward
                reward_to_return = self.LAP_REWARD # TODO: Figure out a good scale here... number of steps required per lap will vary depending on the course; don't want negative reward for completing a lap
            progress = self._get_progress()
            reward_factor = self.PROGRESS_REWARD if progress >= 0 else self.BACKWARDS_PUNISHMENT
            reward_to_return += progress * reward_factor + self.DEFAULT_STEP_REWARD
        self.last_known_lap = cur_lap
        # print("reward:", reward_to_return)
        # if reward_to_return > 1000:
        #     print("whaaa?")
        return reward_to_return

    def _get_lap(self):
        # The first checkpoint is the upper left corner. It's value should tell us the lap.
        ckpt_val = self._evaluate_checkpoint((self.CHECKPOINT_LOCATIONS[0], self.CHECKPOINT_LOCATIONS[1]))

        # If it is unknown, assume same lap (character icon is likely covering the corner)
        return ckpt_val if ckpt_val != -1 else self.lap

    def _generate_checkpoints(self, min_x, min_y, max_x, max_y):
        # TODO: I'm sure this can/should be more pythonic somehow

        # Sample 4 pixels for each checkpoint to reduce the
        # likelihood of a pixel matching the color by chance
        checkpoints = (
            [(min_x + i, min_y) for i in range(max_x - min_x)] + # Top
            [(max_x, min_y + i) for i in range(max_y - min_y)] + # Right
            [(max_x - i, max_y) for i in range(1, max_x - min_x)] + # Bottom, for some reason the bottom right pixel in the progress bar is not rendered
            [(min_x, max_y - i) for i in range(max_y - min_y)]   # Left
        )
        return checkpoints
        

    def _get_current_checkpoint(self):
        checkpoint_values = [self._evaluate_checkpoint(points)
                             for points in self.CHECKPOINT_LOCATIONS]

        # Check if we have achieved any checkpoints
        if any(val > -1 for val in checkpoint_values):
            
            # argmin tells us the first index with the lowest value
            index_of_lowest_val = np.argmin(checkpoint_values)

            if index_of_lowest_val != 0:
                # If the argmin is anything but 0, we have achieved
                # all the checkpoints up through the prior index
                checkpoint = index_of_lowest_val - 1
            else:
                # If the argmin is at index 0, they are all the same value,
                # which means we've hit all the checkpoints for this lap
                checkpoint = len(checkpoint_values) - 1
            
            #if self.last_known_ckpt != checkpoint:
            #    cprint('--------------------------------------------','red')
            #    cprint('Checkpoints: %s' % checkpoint_values, 'yellow')
            #    cprint('Checkpoint: %s' % checkpoint, 'cyan')

            return checkpoint
        else:
            # We haven't hit any checkpoint yet :(
            return -1

    # https://stackoverflow.com/a/3844948
    # Efficiently determines if all items in a list are equal by 
    # counting the occurrences of the first item in the list and 
    # checking if the count matches the length of the list:
    def all_equal(self, some_list):
        return some_list.count(some_list[0]) == len(some_list)

    def _evaluate_checkpoint(self, checkpoint_points):
        checkpoint_pixels = [IMAGE_HELPER.GetPixelColor(self.pixel_array, point[0], point[1])
                             for point in checkpoint_points]
        # print("checkpoint values:", checkpoint_pixels)

        #print(checkpoint_pixels)
        
        # If the first pixel is not a valid color, no need to check the other three
        if not checkpoint_pixels[0] in self.HUD_PROGRESS_COLOR_VALUES:
            return -1
        # print("first is in")
        # If the first pixel is good, make sure the other three match
        if not self.all_equal(checkpoint_pixels):
            return -1
        # print("all equal")
        # If all are good, return the corresponding value
        return self.HUD_PROGRESS_COLOR_VALUES[checkpoint_pixels[0]]

    def _is_stuck(self):
        '''If progress of last x steps is smaller than treshhold, we are stuck'''
        if len(self._last_progresses) < self.AMOUNT_STEPS_CONSIDERED_STUCK:
            return False
        return (sum(self._last_progresses) / len(self._last_progresses))- min(self._last_progresses) <= self.MIN_PROGRESS

    def _went_backwards(self):
        return not all(self._last_progresses[i] <= self._last_progresses[i+1] for i in range(len(self._last_progresses) - 1))

    def _evaluate_end_state(self):
        # print(self._is_stuck())
        # print(self._last_progresses)
        abort_episode = self._is_stuck() or self._went_backwards()
        end_pixel = self.END_PIXELS[self.res_w]
        completed_episode = self.end_race_pixel_color == IMAGE_HELPER.GetPixelColor(self.pixel_array, *end_pixel) #TODO: adjust for smaller resolutions
        return completed_episode, abort_episode

    def _navigate_menu(self):
        self._wait(count=10, wait_for='Nintendo screen')
        self._press_button(ControllerState.A_BUTTON)

        self._wait(count=68, wait_for='Mario Kart splash screen')
        self._press_button(ControllerState.A_BUTTON)

        self._wait(count=68, wait_for='Game Select screen')
        self._navigate_game_select()

        self._wait(count=14, wait_for='Player Select screen')
        self._navigate_player_select()

        self._wait(count=31, wait_for='Map Select screen')
        self._navigate_map_select()

        self._wait(count=46, wait_for='race to load')
        
        # Change HUD View twice to get to the one we want:
        self._cycle_hud_view(times=2)

        # Now that we have the HUD as needed, reset the race so we have a consistent starting frame:
        self._reset_during_race()

    def _navigate_game_select(self):
        # Select number of players (1 player highlighted by default)
        self._press_button(ControllerState.A_BUTTON)
        self._wait(count=3, wait_for='animation')

        # Select GrandPrix or TimeTrials (GrandPrix highlighted by default - down to switch to TimeTrials)
        self._press_button(ControllerState.JOYSTICK_DOWN)
        self._wait(count=3, wait_for='animation')

        # Select TimeTrials
        self._press_button(ControllerState.A_BUTTON)

        # Select Begin
        self._press_button(ControllerState.A_BUTTON)

        # Press OK
        self._press_button(ControllerState.A_BUTTON)

    def _navigate_player_select(self):
        print('Player row: ' + str(self.PLAYER_ROW))
        print('Player col: ' + str(self.PLAYER_COL))

        # Character selection is remembered each time, so ensure upper-left-most is selected
        self._press_button(ControllerState.JOYSTICK_UP)
        self._press_button(ControllerState.JOYSTICK_LEFT, times=3)

        # Navigate to character
        self._press_button(ControllerState.JOYSTICK_DOWN, times=self.PLAYER_ROW)
        self._press_button(ControllerState.JOYSTICK_RIGHT, times=self.PLAYER_COL)

        # Select character
        self._press_button(ControllerState.A_BUTTON)

        # Press OK
        self._press_button(ControllerState.A_BUTTON)

    def _navigate_map_select(self):
        print('Map series: ' + str(self.MAP_SERIES))
        print('Map choice: ' + str(self.MAP_CHOICE))

        # Map series selection is remembered each time, so ensure left-most is selected
        self._press_button(ControllerState.JOYSTICK_LEFT, times=3)

        # Select map series
        self._press_button(ControllerState.JOYSTICK_RIGHT, times=self.MAP_SERIES)
        self._press_button(ControllerState.A_BUTTON)

        # Map choice selection is remembered each time, so ensure top-most is selected
        self._press_button(ControllerState.JOYSTICK_UP, times=3)

        # Select map choice
        self._press_button(ControllerState.JOYSTICK_DOWN, times=self.MAP_CHOICE)
        self._press_button(ControllerState.A_BUTTON)

        # Press OK
        self._press_button(ControllerState.A_BUTTON)

    def _cycle_hud_view(self, times=1):
        for _ in itertools.repeat(None, times):
            self._press_button(ControllerState.CR_BUTTON)

    def _navigate_post_race_menu(self):
        # Times screen
        self._press_button(ControllerState.A_BUTTON)
        self._wait(count=13, wait_for='Post race menu')

        # Post race menu (previous choice selected by default)
        # - Retry
        # - Course Change
        # - Driver Change
        # - Quit
        # - Replay
        # - Save Ghost

        # Because the previous choice is selected by default, we navigate to the top entry so our
        # navigation is consistent. The menu doesn't cycle top to bottom or bottom to top, so we can
        # just make sure we're at the top by hitting up a few times
        self._press_button(ControllerState.JOYSTICK_UP, times=5)

        # Now we are sure to have the top entry selected
        # Go down to 'course change'
        self._press_button(ControllerState.JOYSTICK_DOWN)
        self._press_button(ControllerState.A_BUTTON)

    def _set_character(self, character):
        cprint(f"using character {character}!", "green")
        self.PLAYER_ROW, self.PLAYER_COL = self.CHARACTERS[character]

    def _set_course(self, course):
        cprint(f"playing course {course}!", "green")
        self.MAP_SERIES, self.MAP_CHOICE = self.COURSES[course]
