for($i=0;$i -lt 10;$i++)
{
	$StopWatchJulia = [system.diagnostics.stopwatch]::startNew()
	julia C:\Users\labelname\Documents\neural-networks-computational-macroeconomics\Julia_LSTM_Script.jl
	$StopWatchJulia.Stop()
	$StopWatchJulia.Elapsed.TotalMilliseconds
}