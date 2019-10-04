$scriptPath = Split-Path -parent $PSCommandPath;
$algoPath = "algo_strategy_gene2.py"

py -3 $algoPath
