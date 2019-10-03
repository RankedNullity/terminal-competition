$scriptPath = Split-Path -parent $PSCommandPath;
$algoPath = "algo_strategy.py"

py -3 $algoPath
