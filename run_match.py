import os
import subprocess
import sys

# Runs a single game
def run_single_game(process_command):
    print("Start run a match")
    p = subprocess.Popen(
        process_command,
        shell=True,
        stdout=sys.stdout,
        stderr=sys.stderr
        )
    # daemon necessary so game shuts down if this script is shut down by user
    p.daemon = 1
    p.wait()
    print("Finished running match")

# Get location of this run file
file_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.join(file_dir, os.pardir)
parent_dir = os.path.abspath(parent_dir)

# Get if running in windows OS
is_windows = sys.platform.startswith('win')
print("Is windows: {}".format(is_windows))

# Set default path for algos if script is run with no params
default_algo = ".\\run_starter_strat.ps1" if is_windows else file_dir + "run.sh"
algo1 = default_algo
algo2 = default_algo

# If script run with params, use those algo locations when running the game
if len(sys.argv) > 1:
    algo1 = sys.argv[1]
if len(sys.argv) > 2:
    algo2 = sys.argv[2]

algo1 = ".\\run_starter_strat.ps1"
algo2 = ".\\run_PPO_strat.ps1"

run_single_game("cd {} && java -jar engine.jar work {} {}".format(file_dir, algo1, algo2))