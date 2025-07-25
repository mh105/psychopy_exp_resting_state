# Resting state task
Last edit: 07/17/2025

## Edit history
- 07/17/2025 by Alex He - enabled flexible connection to both Cedrus C-POD and M-POD
- 03/10/2025 by Alex He - added a blank text of 0.3s to make eyes open period duration between triggers more precise
- 11/22/2024 by Alex He - removed summary csv saving since no trialList used
- 10/24/2024 by Alex He - added a print message of task ID at the onset of task
- 10/12/2024 by Alex He - increased logging granularity from warning to debug (maximal level)
- 10/10/2024 by Alex He - added MilliKey response box and finalized voice-over audio
- 09/25/2024 by Alex He - added winHandle.activate() to make sure window is on foreground
- 09/23/2024 by Alex He - upgraded to run on PsychoPy 2024.2.2
- 09/05/2024 by Alex He - removed git tracking of _lastrun.py file and added retries to pyxid2.get_xid_devices() with timeout
- 08/17/2024 by Alex He - added more print messages during c-pod connection
- 08/12/2024 by Alex He - reverted to python 3.8 as pylink connection to EyeLink does not work correctly on 3.10
- 08/04/2024 by Alex He - generated experiment scripts on python 3.10
- 08/02/2024 by Alex He - upgraded to support PsychoPy 2024.2.1
- 07/23/2024 by Alex He - upgraded to support PsychoPy 2024.2.0
- 07/09/2024 by Alex He - created finalized first draft version

## Description
This is a resting state used to measure occipital alpha activity in human subjects. The task begins with a 3-min eyes-closed period, during which occipital alpha rhythm is expected to be strong and robust. Then the subject is instructed to look at a fixation cross at the center of the screen for another 3-min period, during which the occipital alpha activity is expected to attenuate in strength.

## Outcome measures
- Occipital alpha strength (mean spectral power / magnitude)
- Individual alpha frequency
- Alpha reactivity to visual inputs (eyes-closed minus eyes-open)
- Alpha stability quantified with switching state-space modeling
