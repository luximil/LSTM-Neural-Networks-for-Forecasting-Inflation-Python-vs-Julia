for($i=0;$i -lt 10;$i++)
{
	$StopWatchPython = [system.diagnostics.stopwatch]::startNew()
	python C:\Users\labelname\Documents\neural-networks-computational-macroeconomics\Python_LSTM_Script.py
	$StopWatchPython.Stop()
	$StopWatchPython.Elapsed.TotalMilliseconds
}