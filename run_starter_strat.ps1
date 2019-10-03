$scriptPath = Split-Path -parent $PSCommandPath;
$algoPath = "algo_strategy_starter.py"

py -3 $algoPath
