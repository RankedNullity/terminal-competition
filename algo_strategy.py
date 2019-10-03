import gamelib
import random
import math
import warnings
from sys import maxsize
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import PPO.ActorCritic
from random import choices

"""
Most of the algo code you write will be in this file unless you create new
modules yourself. Start by modifying the 'on_turn' function.

Advanced strategy tips: 

  - You can analyze action frames by modifying on_action_frame function

  - The GameState.map object can be manually manipulated to create hypothetical 
  board states. Though, we recommended making a copy of the map to preserve 
  the actual current map state.
"""

class AlgoStrategy(gamelib.AlgoCore):
    def __init__(self):
        super().__init__()
        seed = random.randrange(maxsize)
        random.seed(seed)
        gamelib.debug_write('Random seed: {}'.format(seed))
        self.model = ActorCritic()
        # TODO: Specificy file_path
        self.model.load_state_dict(torch.load(file_path))
        self.actions = []
        
    def on_game_start(self, config):
        """ 
        Read in config and perform any initial setup here 
        """
        gamelib.debug_write('Configuring your custom algo strategy...')
        self.config = config
        global FILTER, ENCRYPTOR, DESTRUCTOR, PING, EMP, SCRAMBLER, PIECE_TO_INT, INT_TO_PIECE
        FILTER = config["unitInformation"][0]["shorthand"]
        ENCRYPTOR = config["unitInformation"][1]["shorthand"]
        DESTRUCTOR = config["unitInformation"][2]["shorthand"]
        PING = config["unitInformation"][3]["shorthand"]
        EMP = config["unitInformation"][4]["shorthand"]
        SCRAMBLER = config["unitInformation"][5]["shorthand"]
        # This is a good place to do initial setup
        self.scored_on_locations = []

        PIECE_TO_INT = {FILTER: 1, ENCRYPTOR: 2, DESTRUCTOR: 3, PING: 4, EMP: 5, SCRAMBLER: 6}
        INT_TO_PIECE = {1: FILTER, 2: ENCRYPTOR, 3: DESTRUCTOR, 4: PING, 5: EMP, 6: SCRAMBLER }
    
    def parse_serialized_string(serialized_string):
        state = json.loads(serialized_string)
        turn_info = state["turnInfo"]
        
        # Input game state as a matrix.
        # Channel 0: Piece type (0) for empty, (1, 2, 3) for stationary, (4,5,6) for moving pieces
        # Channel 1: Piece HP
        # Channel 2: Piece Side [0 = empty, 1 = Friendly, 2 = Enemy]
        # Channel 3: Amount of Pieces (-1 = marked for removal)
        board = np.zeros((28, 28, 4))
        p1units = state["p1Units"]
        p2units = state["p2Units"]
        update_board(board, p1units, 0)
        update_board(board, p2units, 1)

        # Formats relevant game information in gamedata
        gamedata = np.zeros(7)
        p1_health, p1_cores, p1_bits, p1_time = map(float, state["p1Stats"][:4])
        p2_health, p2_cores, p2_bits, p2_time = map(float, state["p2Stats"][:4])
        gamedata[0] = game_state.p1_health
        gamedata[1] = game_state.p1_cores
        gamedata[2] = game_state.p1_bits
        gamedata[3] = game_state.p2_health
        gamedata[4] = game_state.p2_cores
        gamedata[5] = game_state.p2_bits
        gamedata[6] = int(turn_info[1])
        return board, gamedata
        
    def update_board(board, units, player_number):
        typedef = self.config.get("unitInformation")
        for i, unit_types in enumerate(units):
            for uinfo in unit_types:
                unit_type = typedef[i].get("shorthand")
                sx, sy, shp = uinfo[:3]
                x, y = map(int, [sx, sy])
                hp = float(shp)
                if board[x, y, 0] == 0:
                    board[x, y, 0] = i + 1
                    board[x, y, 1] = hp
                    board[x, y, 2] = player_number + 1
                board[x, y, 3] += 1
                
                # This depends on RM always being the last type to be processed
                if unit_type == typedef[6]["shorthand"]:
                     board[x, y, 3] = -1
        
    def parse_gamestate(game_state):
        board_width, board_height = 28, 28
        game_map = game_state.game_map
        # Input game state as a matrix.
        # Channel 0: Piece type (0) for empty, (1, 2, 3) for stationary, (4,5,6) for moving pieces
        # Channel 1: Piece HP
        # Channel 2: Piece Side [0 = empty, 1 = Friendly, 2 = Enemy]
        # Channel 3: Amount of Pieces (-1 = marked for removal)
        board = np.zeros((board_width, board_height, 4))
        for i in range(board_width):
            for j in range(board_height):
                current_list = game_map[i, j]

                if len(current_list) != 0:
                    hp_total = 0
                    for unit in current_list:
                        hp_total += unit.stability
                    # Assign the correct unit type
                    unit = current_list[0]
                    board[i, j, 1] = hp_total
                    board[i, j, 2] = unit.player_index + 1
                    board[i, j, 3] = len(current_list)
                    board[i, j, 0] = PIECE_TO_INT[unit.unit_type]        
                    
        # HP, Cores, Bits of both players.
        gamedata = np.zeros(7)
        gamedata[0] = game_state.my_health
        gamedata[1] = game_state._player_resources[0]['cores']
        gamedata[2] = game_state._player_resources[0]['bits']
        gamedata[3] = game_state.enemy_health
        gamedata[4] = game_state._player_resources[1]['cores']
        gamedata[5] = game_state._player_resources[1]['bits']
        metadata[6] = game_state.turn_number
        return board, gamedata
                                                      
    
    def perform_action_using_output(output, game_state):
        '''Performs an action using the output of the PPO network, and submits using game_state'''
        # moveboard 28 x 14 half of the board [0: Num of units placed, 1: type of unit placed]
        move_board = np.zeros(28, 14, 2)
        
        # Assume output is 14x14x 18
        row_cutoffs = [x for x in range(28,0,-2)]
        cum_row_cutoff = [sum(row_cuttoffs[0:x]) for x in range(len(row_cutoffs))]
        rowNum = 0
        for i in range(14 * 15):
            if i >= cum_row_cutoff[rowNum]:
                rowNum += 1
            rowPos = i - cum_row_cutoff[rowNum]
            x = rowNum + rowPos
            y = 13 - rowNum
            for j in range(18):
                index = i * 18 + j
                num_placement_probs = softmax(output[index:index + 12])
                chosen_num = choices(np.arange(-1, 11), num_placement_probs)

                if chosen_num == -1:
                    game_state.attempt_remove(x, y)
                    move_board[x, y, 0] = -1
                elif not chosen_num == 0:
                    index += 12
                    if game_state.can_spawn(PING, (x,y)):
                        # sample from all 6 and choose unit type.
                        piece_type_probs = softmax(output[index: index + 6])
                        chosen_type = choices(np.arange(1,7), piece_type_probs)
                        true_num = chosen_num if chosen_num > 3 else 1
                        game_state.attempt_spawn(INT_TO_PIECE(chosen_type), (x,y), true_num)
                        move_board[x, y, 0] = true_num
                        move_board[x, y, 1] = chosen_type
                    else:
                        # sample from only the first 3                           
                        piece_type_probs = softmax(output[index: index + 3])
                        chosen_type = choices(np.arange(1,4), piece_type_probs)
                        game_state.attempt_spawn(INT_TO_PIECE(chosen_type), (x,y))
                        move_board[x, y, 0] = true_num
                        move_board[x, y, 1] = chosen_type

        self.actions.append(move_board)

    def softmax(x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def on_turn(self, turn_state):
        """
        This function is called every turn with the game state wrapper as
        an argument. The wrapper stores the state of the arena and has methods
        for querying its state, allocating your current resources as planned
        unit deployments, and transmitting your intended deployments to the
        game engine.
        """
        game_state = gamelib.GameState(self.config, turn_state)
        gamelib.debug_write('Performing turn {} of your custom algo strategy'.format(game_state.turn_number))
        game_state.suppress_warnings(True)  #Comment or remove this line to enable warnings.

        board_state, game_data = parse_gamestate(game_state)
        network_output = self.model.forward(board_state, game_data)
        perform_action_using_output(network_output, game_state)
        game_state.submit_turn()


        
    def filter_blocked_locations(self, locations, game_state):
        filtered = []
        for location in locations:
            if not game_state.contains_stationary_unit(location):
                filtered.append(location)
        return filtered

    def on_action_frame(self, turn_string):
        """
        This is the action frame of the game. This function could be called 
        hundreds of times per turn and could slow the algo down so avoid putting slow code here.
        Processing the action frames is complicated so we only suggest it if you have time and experience.
        Full doc on format of a game frame at: https://docs.c1games.com/json-docs.html
        """
        # Let's record at what position we get scored on
        state = json.loads(turn_string)
        events = state["events"]
        breaches = events["breach"]
        for breach in breaches:
            location = breach[0]
            unit_owner_self = True if breach[4] == 1 else False
            # When parsing the frame data directly, 
            # 1 is integer for yourself, 2 is opponent (StarterKit code uses 0, 1 as player_index instead)
            if not unit_owner_self:
                gamelib.debug_write("Got scored on at: {}".format(location))
                self.scored_on_locations.append(location)
                gamelib.debug_write("All locations: {}".format(self.scored_on_locations))


if __name__ == "__main__":
    algo = AlgoStrategy()
    algo.start()
