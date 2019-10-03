$scriptPath = Split-Path -parent $PSCommandPath;
$algoPath = "algo_strategy_ppo.py"

py -3 $algoPath
