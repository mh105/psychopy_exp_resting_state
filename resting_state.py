#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2024.2.0dev2),
    on Sat Jun 15 11:54:11 2024
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
prefs.hardware['audioLatencyMode'] = '4'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware, iohub
from psychopy.tools import environmenttools
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER, priority)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

import psychopy.iohub as io
from psychopy.hardware import keyboard

# Run 'Before Experiment' code from eeg
import pyxid2

devices = pyxid2.get_xid_devices()

if devices:
    dev = devices[0]
    assert dev.device_name == 'Cedrus C-POD', "Incorrect C-POD detected."
    dev.set_pulse_duration(50)  # set pulse duration to 50ms

    # Start EEG recording
    dev.activate_line(bitmask=126)  # trigger 126 will start EEG
    core.wait(10)  # wait 10s for the EEG system to start recording

    # Marching lights test
    print("C-POD<->eego 7-bit trigger lines test...")
    for line in range(1, 8):  # raise lines 1-7 one at a time
        print("  raising line {} (bitmask {})".format(line, 2 ** (line-1)))
        dev.activate_line(lines=line)
        core.wait(0.5)  # wait 500ms between two consecutive triggers
    dev.con.set_digio_lines_to_mask(0)  # XidDevice.clear_all_lines()

else:
    # Dummy XidDevice for code components to run without C-POD connected
    class dummyXidDevice(object):
        def __init__(self):
            pass
        def activate_line(self, lines=None, bitmask=None):
            pass


    print("WARNING: No C-POD connected for this session! "
          "You must start/stop EEG recording manually!")
    dev = dummyXidDevice()

# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2024.2.0dev2'
expName = 'resting_state'  # from the Builder filename that created this script
# information about this experiment
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'session': '001',
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = True
_winSize = [2560, 1440]
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version='',
        extraInfo=expInfo, runtimeInfo=None,
        originPath='/Users/alexhe/Dropbox (Personal)/Active_projects/PsychoPy/exp_resting_state/resting_state.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # set how much information should be printed to the console / app
    if PILOTING:
        logging.console.setLevel(
            prefs.piloting['pilotConsoleLoggingLevel']
        )
    else:
        logging.console.setLevel('warning')
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log')
    if PILOTING:
        logFile.setLevel(
            prefs.piloting['pilotLoggingLevel']
        )
    else:
        logFile.setLevel(
            logging.getLevel('warning')
        )
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if PILOTING:
        logging.debug('Fullscreen settings ignored as running in pilot mode.')
    
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=0,
            winType='pyglet', allowStencil=False,
            monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height', 
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0,0,0]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    win.mouseVisible = False
    win.hideMessage()
    # show a visual indicator if we're in piloting mode
    if PILOTING and prefs.piloting['showPilotingIndicator']:
        win.showPilotingIndicator()
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    
    # Setup eyetracking
    ioConfig['eyetracker.hw.sr_research.eyelink.EyeTracker'] = {
        'name': 'tracker',
        'model_name': 'EYELINK 1000 DESKTOP',
        'simulation_mode': False,
        'network_settings': '100.1.1.1',
        'default_native_data_file_name': 'EXPFILE',
        'runtime_settings': {
            'sampling_rate': 1000.0,
            'track_eyes': 'LEFT_EYE',
            'sample_filtering': {
                'sample_filtering': 'FILTER_LEVEL_OFF',
                'elLiveFiltering': 'FILTER_LEVEL_OFF',
            },
            'vog_settings': {
                'pupil_measure_types': 'PUPIL_DIAMETER',
                'tracking_mode': 'PUPIL_CR_TRACKING',
                'pupil_center_algorithm': 'ELLIPSE_FIT',
            }
        }
    }
    
    # Setup iohub keyboard
    ioConfig['Keyboard'] = dict(use_keymap='psychopy')
    
    ioSession = '1'
    if 'session' in expInfo:
        ioSession = str(expInfo['session'])
    ioServer = io.launchHubServer(window=win, **ioConfig)
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    deviceManager.devices['eyetracker'] = ioServer.getDevice('tracker')
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='iohub'
        )
    if deviceManager.getDevice('key_welcome') is None:
        # initialise key_welcome
        key_welcome = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_welcome',
        )
    # create speaker 'read_welcome'
    deviceManager.addDevice(
        deviceName='read_welcome',
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        index=-1
    )
    if deviceManager.getDevice('key_et') is None:
        # initialise key_et
        key_et = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_et',
        )
    # create speaker 'read_et'
    deviceManager.addDevice(
        deviceName='read_et',
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        index=-1
    )
    # create speaker 'read_start'
    deviceManager.addDevice(
        deviceName='read_start',
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        index=-1
    )
    if deviceManager.getDevice('key_instruct') is None:
        # initialise key_instruct
        key_instruct = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_instruct',
        )
    # create speaker 'read_instruct'
    deviceManager.addDevice(
        deviceName='read_instruct',
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        index=-1
    )
    # create speaker 'tone_finish'
    deviceManager.addDevice(
        deviceName='tone_finish',
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        index=-1
    )
    if deviceManager.getDevice('key_instruct_2') is None:
        # initialise key_instruct_2
        key_instruct_2 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_instruct_2',
        )
    # create speaker 'read_instruct_2'
    deviceManager.addDevice(
        deviceName='read_instruct_2',
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        index=-1
    )
    # create speaker 'read_thank_you'
    deviceManager.addDevice(
        deviceName='read_thank_you',
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        index=-1
    )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], playbackComponents=[]):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    playbackComponents : list, tuple
        List of any components with a `pause` method which need to be paused.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # pause any playback components
    for comp in playbackComponents:
        comp.pause()
    # prevent components from auto-drawing
    win.stashAutoDraw()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='ioHub',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # flip the screen
        win.flip()
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    for comp in playbackComponents:
        comp.play()
    # restore auto-drawn components
    win.retrieveAutoDraw()
    # reset any timers
    for timer in timers:
        timer.reset()


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ioHub'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "_welcome" ---
    text_welcome = visual.TextStim(win=win, name='text_welcome',
        text='Welcome! This task will take approximately 7 minutes.\n\nBefore we explain the task, we need to first calibrate the eyetracking camera. Please sit in a comfortable position with your head on the chin rest. Once we begin, it is important that you stay in the same position throughout this task.\n\nPlease take a moment to adjust the chair height, chin rest, and sitting posture. Make sure that you feel comfortable and can stay still for a while.\n\n\nWhen you are ready, press the spacebar',
        font='Arial',
        units='norm', pos=(0, 0), draggable=False, height=0.1, wrapWidth=1.8, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_welcome = keyboard.Keyboard(deviceName='key_welcome')
    read_welcome = sound.Sound(
        'A', 
        secs=-1, 
        stereo=True, 
        hamming=True, 
        speaker='read_welcome',    name='read_welcome'
    )
    read_welcome.setVolume(1.0)
    
    # --- Initialize components for Routine "_et_instruct" ---
    text_et = visual.TextStim(win=win, name='text_et',
        text='During the calibration, you will see a target circle moving around the screen. Please try to track it with your eyes.\n\nMake sure to keep looking at the circle when it stops, and follow it when it moves. It is important that you keep your head on the chin rest once this part begins.\n\n\nPress the spacebar when you are ready, and our team will start the calibration for you',
        font='Arial',
        units='norm', pos=(0, 0), draggable=False, height=0.1, wrapWidth=1.8, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_et = keyboard.Keyboard(deviceName='key_et')
    read_et = sound.Sound(
        'A', 
        secs=-1, 
        stereo=True, 
        hamming=True, 
        speaker='read_et',    name='read_et'
    )
    read_et.setVolume(1.0)
    
    # --- Initialize components for Routine "_et_mask" ---
    text_mask = visual.TextStim(win=win, name='text_mask',
        text=None,
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "__start__" ---
    text_start = visual.TextStim(win=win, name='text_start',
        text='We are now ready to begin...',
        font='Arial',
        units='norm', pos=(0, 0), draggable=False, height=0.1, wrapWidth=1.8, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    read_start = sound.Sound(
        'A', 
        secs=1.6, 
        stereo=True, 
        hamming=True, 
        speaker='read_start',    name='read_start'
    )
    read_start.setVolume(1.0)
    # Run 'Begin Experiment' code from trigger_table
    ##TASK ID TRIGGER VALUES##
    # special code 100 (task start, task ID should follow immediately)
    task_start_code = 100
    # special code 101 (task ID for 3min EC + 3min EO resting state task)
    task_ID_code = 101
    
    ##GENERAL TRIGGER VALUES##
    # special code 122 (block start)
    block_start_code = 122
    # special code 123 (block end)
    block_end_code = 123
    
    ##TASK SPECIFIC TRIGGER VALUES##
    # N.B.: only use values 1-99 and provide clear comments on used values
    
    # Run 'Begin Experiment' code from task_id
    dev.activate_line(bitmask=task_start_code)  # special code for task start
    core.wait(0.5)  # wait 500ms between two consecutive triggers
    dev.activate_line(bitmask=task_ID_code)  # special code for task ID
    
    etRecord = hardware.eyetracker.EyetrackerControl(
        tracker=eyetracker,
        actionType='Start Only'
    )
    
    # --- Initialize components for Routine "instruct_ec" ---
    text_instruct = visual.TextStim(win=win, name='text_instruct',
        text='This is a resting task. It is designed to measure brain waves when we are not thinking hard or trying to do anything.\n\nFor the next 3 minutes, please close your eyes, and relax. Just let your mind wander. Try not to move, and try to stay awake.\n\nWhen the 3-minute is up, you will hear a tone to indicate the completion.\n\n\nPress the spacebar to begin',
        font='Arial',
        units='norm', pos=(0, 0), draggable=False, height=0.1, wrapWidth=1.8, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_instruct = keyboard.Keyboard(deviceName='key_instruct')
    read_instruct = sound.Sound(
        'A', 
        secs=-1, 
        stereo=True, 
        hamming=True, 
        speaker='read_instruct',    name='read_instruct'
    )
    read_instruct.setVolume(1.0)
    
    # --- Initialize components for Routine "eyes_closed" ---
    text_ec = visual.TextStim(win=win, name='text_ec',
        text=None,
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    tone_finish = sound.Sound(
        'A', 
        secs=0.3, 
        stereo=True, 
        hamming=True, 
        speaker='tone_finish',    name='tone_finish'
    )
    tone_finish.setVolume(1.0)
    
    # --- Initialize components for Routine "instruct_eo" ---
    text_instruct_2 = visual.TextStim(win=win, name='text_instruct_2',
        text='Great! That is the first of two parts in this task.\n\nFor the next 3 minutes, please look at the fixation cross on the screen, and relax. As before, just let your mind wander. You may blink normally, but try to keep your eyes open, and try not to move.\n\nWhen the 3-minute is up, the cross will disappear to indicate the completion.\n\n\nPress the spacebar to begin',
        font='Arial',
        units='norm', pos=(0, 0), draggable=False, height=0.1, wrapWidth=1.8, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_instruct_2 = keyboard.Keyboard(deviceName='key_instruct_2')
    read_instruct_2 = sound.Sound(
        'A', 
        secs=-1, 
        stereo=True, 
        hamming=True, 
        speaker='read_instruct_2',    name='read_instruct_2'
    )
    read_instruct_2.setVolume(1.0)
    
    # --- Initialize components for Routine "eyes_open" ---
    text_eo = visual.TextStim(win=win, name='text_eo',
        text='+',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.1, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "_thank_you" ---
    text_thank_you = visual.TextStim(win=win, name='text_thank_you',
        text='Thank you. You have completed this task!',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    read_thank_you = sound.Sound(
        'A', 
        secs=-1, 
        stereo=True, 
        hamming=True, 
        speaker='read_thank_you',    name='read_thank_you'
    )
    read_thank_you.setVolume(1.0)
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # --- Prepare to start Routine "_welcome" ---
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_welcome
    key_welcome.keys = []
    key_welcome.rt = []
    _key_welcome_allKeys = []
    read_welcome.setSound('resource/welcome.wav', hamming=True)
    read_welcome.setVolume(1.0, log=False)
    read_welcome.seek(0)
    # keep track of which components have finished
    _welcomeComponents = [text_welcome, key_welcome, read_welcome]
    for thisComponent in _welcomeComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "_welcome" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_welcome* updates
        
        # if text_welcome is starting this frame...
        if text_welcome.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_welcome.frameNStart = frameN  # exact frame index
            text_welcome.tStart = t  # local t and not account for scr refresh
            text_welcome.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_welcome, 'tStartRefresh')  # time at next scr refresh
            # update status
            text_welcome.status = STARTED
            text_welcome.setAutoDraw(True)
        
        # if text_welcome is active this frame...
        if text_welcome.status == STARTED:
            # update params
            pass
        
        # *key_welcome* updates
        waitOnFlip = False
        
        # if key_welcome is starting this frame...
        if key_welcome.status == NOT_STARTED and tThisFlip >= 0.2-frameTolerance:
            # keep track of start time/frame for later
            key_welcome.frameNStart = frameN  # exact frame index
            key_welcome.tStart = t  # local t and not account for scr refresh
            key_welcome.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_welcome, 'tStartRefresh')  # time at next scr refresh
            # update status
            key_welcome.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_welcome.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_welcome.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_welcome.status == STARTED and not waitOnFlip:
            theseKeys = key_welcome.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=True)
            _key_welcome_allKeys.extend(theseKeys)
            if len(_key_welcome_allKeys):
                key_welcome.keys = _key_welcome_allKeys[0].name  # just the first key pressed
                key_welcome.rt = _key_welcome_allKeys[0].rt
                key_welcome.duration = _key_welcome_allKeys[0].duration
                # a response ends the routine
                continueRoutine = False
        
        # if read_welcome is starting this frame...
        if read_welcome.status == NOT_STARTED and tThisFlip >= 0.8-frameTolerance:
            # keep track of start time/frame for later
            read_welcome.frameNStart = frameN  # exact frame index
            read_welcome.tStart = t  # local t and not account for scr refresh
            read_welcome.tStartRefresh = tThisFlipGlobal  # on global time
            # update status
            read_welcome.status = STARTED
            read_welcome.play(when=win)  # sync with win flip
        
        # if read_welcome is active this frame...
        if read_welcome.status == STARTED:
            # update params
            pass
            # if sound is finished but read_welcome has time left, finish it now
            if read_welcome.isFinished:
                read_welcome.status = FINISHED
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in _welcomeComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "_welcome" ---
    for thisComponent in _welcomeComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # check responses
    if key_welcome.keys in ['', [], None]:  # No response was made
        key_welcome.keys = None
    thisExp.addData('key_welcome.keys',key_welcome.keys)
    if key_welcome.keys != None:  # we had a response
        thisExp.addData('key_welcome.rt', key_welcome.rt)
        thisExp.addData('key_welcome.duration', key_welcome.duration)
    read_welcome.pause()  # ensure sound has stopped at end of Routine
    thisExp.nextEntry()
    # the Routine "_welcome" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "_et_instruct" ---
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_et
    key_et.keys = []
    key_et.rt = []
    _key_et_allKeys = []
    read_et.setSound('resource/eyetrack_calibrate_instruct.wav', hamming=True)
    read_et.setVolume(1.0, log=False)
    read_et.seek(0)
    # keep track of which components have finished
    _et_instructComponents = [text_et, key_et, read_et]
    for thisComponent in _et_instructComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "_et_instruct" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_et* updates
        
        # if text_et is starting this frame...
        if text_et.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_et.frameNStart = frameN  # exact frame index
            text_et.tStart = t  # local t and not account for scr refresh
            text_et.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_et, 'tStartRefresh')  # time at next scr refresh
            # update status
            text_et.status = STARTED
            text_et.setAutoDraw(True)
        
        # if text_et is active this frame...
        if text_et.status == STARTED:
            # update params
            pass
        
        # *key_et* updates
        waitOnFlip = False
        
        # if key_et is starting this frame...
        if key_et.status == NOT_STARTED and tThisFlip >= 0.2-frameTolerance:
            # keep track of start time/frame for later
            key_et.frameNStart = frameN  # exact frame index
            key_et.tStart = t  # local t and not account for scr refresh
            key_et.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_et, 'tStartRefresh')  # time at next scr refresh
            # update status
            key_et.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_et.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_et.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_et.status == STARTED and not waitOnFlip:
            theseKeys = key_et.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=True)
            _key_et_allKeys.extend(theseKeys)
            if len(_key_et_allKeys):
                key_et.keys = _key_et_allKeys[0].name  # just the first key pressed
                key_et.rt = _key_et_allKeys[0].rt
                key_et.duration = _key_et_allKeys[0].duration
                # a response ends the routine
                continueRoutine = False
        
        # if read_et is starting this frame...
        if read_et.status == NOT_STARTED and tThisFlip >= 0.8-frameTolerance:
            # keep track of start time/frame for later
            read_et.frameNStart = frameN  # exact frame index
            read_et.tStart = t  # local t and not account for scr refresh
            read_et.tStartRefresh = tThisFlipGlobal  # on global time
            # update status
            read_et.status = STARTED
            read_et.play(when=win)  # sync with win flip
        
        # if read_et is active this frame...
        if read_et.status == STARTED:
            # update params
            pass
            # if sound is finished but read_et has time left, finish it now
            if read_et.isFinished:
                read_et.status = FINISHED
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in _et_instructComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "_et_instruct" ---
    for thisComponent in _et_instructComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # check responses
    if key_et.keys in ['', [], None]:  # No response was made
        key_et.keys = None
    thisExp.addData('key_et.keys',key_et.keys)
    if key_et.keys != None:  # we had a response
        thisExp.addData('key_et.rt', key_et.rt)
        thisExp.addData('key_et.duration', key_et.duration)
    read_et.pause()  # ensure sound has stopped at end of Routine
    thisExp.nextEntry()
    # the Routine "_et_instruct" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "_et_mask" ---
    continueRoutine = True
    # update component parameters for each repeat
    # keep track of which components have finished
    _et_maskComponents = [text_mask]
    for thisComponent in _et_maskComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "_et_mask" ---
    routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 0.05:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_mask* updates
        
        # if text_mask is starting this frame...
        if text_mask.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_mask.frameNStart = frameN  # exact frame index
            text_mask.tStart = t  # local t and not account for scr refresh
            text_mask.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_mask, 'tStartRefresh')  # time at next scr refresh
            # update status
            text_mask.status = STARTED
            text_mask.setAutoDraw(True)
        
        # if text_mask is active this frame...
        if text_mask.status == STARTED:
            # update params
            pass
        
        # if text_mask is stopping this frame...
        if text_mask.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text_mask.tStartRefresh + 0.05-frameTolerance:
                # keep track of stop time/frame for later
                text_mask.tStop = t  # not accounting for scr refresh
                text_mask.tStopRefresh = tThisFlipGlobal  # on global time
                text_mask.frameNStop = frameN  # exact frame index
                # update status
                text_mask.status = FINISHED
                text_mask.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in _et_maskComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "_et_mask" ---
    for thisComponent in _et_maskComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-0.050000)
    thisExp.nextEntry()
    # define target for _et_cal
    _et_calTarget = visual.TargetStim(win, 
        name='_et_calTarget',
        radius=0.015, fillColor='white', borderColor='green', lineWidth=2.0,
        innerRadius=0.005, innerFillColor='black', innerBorderColor='black', innerLineWidth=2.0,
        colorSpace='rgb', units=None
    )
    # define parameters for _et_cal
    _et_cal = hardware.eyetracker.EyetrackerCalibration(win, 
        eyetracker, _et_calTarget,
        units=None, colorSpace='rgb',
        progressMode='time', targetDur=1.5, expandScale=1.5,
        targetLayout='NINE_POINTS', randomisePos=True, textColor='white',
        movementAnimation=True, targetDelay=1.0
    )
    # run calibration
    _et_cal.run()
    # clear any keypresses from during _et_cal so they don't interfere with the experiment
    defaultKeyboard.clearEvents()
    thisExp.nextEntry()
    # the Routine "_et_cal" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "__start__" ---
    continueRoutine = True
    # update component parameters for each repeat
    read_start.setSound('resource/ready_to_begin.wav', secs=1.6, hamming=True)
    read_start.setVolume(1.0, log=False)
    read_start.seek(0)
    # keep track of which components have finished
    __start__Components = [text_start, read_start, etRecord]
    for thisComponent in __start__Components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "__start__" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_start* updates
        
        # if text_start is starting this frame...
        if text_start.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_start.frameNStart = frameN  # exact frame index
            text_start.tStart = t  # local t and not account for scr refresh
            text_start.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_start, 'tStartRefresh')  # time at next scr refresh
            # update status
            text_start.status = STARTED
            text_start.setAutoDraw(True)
        
        # if text_start is active this frame...
        if text_start.status == STARTED:
            # update params
            pass
        
        # if text_start is stopping this frame...
        if text_start.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text_start.tStartRefresh + 2.0-frameTolerance:
                # keep track of stop time/frame for later
                text_start.tStop = t  # not accounting for scr refresh
                text_start.tStopRefresh = tThisFlipGlobal  # on global time
                text_start.frameNStop = frameN  # exact frame index
                # update status
                text_start.status = FINISHED
                text_start.setAutoDraw(False)
        
        # if read_start is starting this frame...
        if read_start.status == NOT_STARTED and tThisFlip >= 0.2-frameTolerance:
            # keep track of start time/frame for later
            read_start.frameNStart = frameN  # exact frame index
            read_start.tStart = t  # local t and not account for scr refresh
            read_start.tStartRefresh = tThisFlipGlobal  # on global time
            # update status
            read_start.status = STARTED
            read_start.play(when=win)  # sync with win flip
        
        # if read_start is stopping this frame...
        if read_start.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > read_start.tStartRefresh + 1.6-frameTolerance:
                # keep track of stop time/frame for later
                read_start.tStop = t  # not accounting for scr refresh
                read_start.tStopRefresh = tThisFlipGlobal  # on global time
                read_start.frameNStop = frameN  # exact frame index
                # update status
                read_start.status = FINISHED
                read_start.stop()
        
        # if read_start is active this frame...
        if read_start.status == STARTED:
            # update params
            pass
            # if sound is finished but read_start has time left, finish it now
            if read_start.isFinished:
                read_start.status = FINISHED
        # *etRecord* updates
        
        # if etRecord is starting this frame...
        if etRecord.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            etRecord.frameNStart = frameN  # exact frame index
            etRecord.tStart = t  # local t and not account for scr refresh
            etRecord.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(etRecord, 'tStartRefresh')  # time at next scr refresh
            # update status
            etRecord.status = STARTED
            etRecord.start()
        if etRecord.status == STARTED:
            etRecord.tStop = t  # not accounting for scr refresh
            etRecord.tStopRefresh = tThisFlipGlobal  # on global time
            etRecord.frameNStop = frameN  # exact frame index
            etRecord.status = FINISHED
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in __start__Components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "__start__" ---
    for thisComponent in __start__Components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    read_start.pause()  # ensure sound has stopped at end of Routine
    # make sure the eyetracker recording stops
    if etRecord.status != FINISHED:
        etRecord.status = FINISHED
    thisExp.nextEntry()
    # the Routine "__start__" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "instruct_ec" ---
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_instruct
    key_instruct.keys = []
    key_instruct.rt = []
    _key_instruct_allKeys = []
    read_instruct.setSound('resource/instruct_ec_reading_audio.wav', hamming=True)
    read_instruct.setVolume(1.0, log=False)
    read_instruct.seek(0)
    # keep track of which components have finished
    instruct_ecComponents = [text_instruct, key_instruct, read_instruct]
    for thisComponent in instruct_ecComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "instruct_ec" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_instruct* updates
        
        # if text_instruct is starting this frame...
        if text_instruct.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_instruct.frameNStart = frameN  # exact frame index
            text_instruct.tStart = t  # local t and not account for scr refresh
            text_instruct.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_instruct, 'tStartRefresh')  # time at next scr refresh
            # update status
            text_instruct.status = STARTED
            text_instruct.setAutoDraw(True)
        
        # if text_instruct is active this frame...
        if text_instruct.status == STARTED:
            # update params
            pass
        
        # *key_instruct* updates
        waitOnFlip = False
        
        # if key_instruct is starting this frame...
        if key_instruct.status == NOT_STARTED and tThisFlip >= 0.2-frameTolerance:
            # keep track of start time/frame for later
            key_instruct.frameNStart = frameN  # exact frame index
            key_instruct.tStart = t  # local t and not account for scr refresh
            key_instruct.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_instruct, 'tStartRefresh')  # time at next scr refresh
            # update status
            key_instruct.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_instruct.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_instruct.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_instruct.status == STARTED and not waitOnFlip:
            theseKeys = key_instruct.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=True)
            _key_instruct_allKeys.extend(theseKeys)
            if len(_key_instruct_allKeys):
                key_instruct.keys = _key_instruct_allKeys[0].name  # just the first key pressed
                key_instruct.rt = _key_instruct_allKeys[0].rt
                key_instruct.duration = _key_instruct_allKeys[0].duration
                # a response ends the routine
                continueRoutine = False
        
        # if read_instruct is starting this frame...
        if read_instruct.status == NOT_STARTED and tThisFlip >= 0.8-frameTolerance:
            # keep track of start time/frame for later
            read_instruct.frameNStart = frameN  # exact frame index
            read_instruct.tStart = t  # local t and not account for scr refresh
            read_instruct.tStartRefresh = tThisFlipGlobal  # on global time
            # update status
            read_instruct.status = STARTED
            read_instruct.play(when=win)  # sync with win flip
        
        # if read_instruct is active this frame...
        if read_instruct.status == STARTED:
            # update params
            pass
            # if sound is finished but read_instruct has time left, finish it now
            if read_instruct.isFinished:
                read_instruct.status = FINISHED
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in instruct_ecComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instruct_ec" ---
    for thisComponent in instruct_ecComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # check responses
    if key_instruct.keys in ['', [], None]:  # No response was made
        key_instruct.keys = None
    thisExp.addData('key_instruct.keys',key_instruct.keys)
    if key_instruct.keys != None:  # we had a response
        thisExp.addData('key_instruct.rt', key_instruct.rt)
        thisExp.addData('key_instruct.duration', key_instruct.duration)
    read_instruct.pause()  # ensure sound has stopped at end of Routine
    thisExp.nextEntry()
    # the Routine "instruct_ec" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "eyes_closed" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('eyes_closed.started', globalClock.getTime(format='float'))
    tone_finish.setSound('Bsh', secs=0.3, hamming=True)
    tone_finish.setVolume(1.0, log=False)
    tone_finish.seek(0)
    # Run 'Begin Routine' code from trigger_ec
    stimulus_pulse_started = False
    stimulus_pulse_started_2 = False
    
    # keep track of which components have finished
    eyes_closedComponents = [text_ec, tone_finish]
    for thisComponent in eyes_closedComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "eyes_closed" ---
    routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 180.3:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_ec* updates
        
        # if text_ec is starting this frame...
        if text_ec.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_ec.frameNStart = frameN  # exact frame index
            text_ec.tStart = t  # local t and not account for scr refresh
            text_ec.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_ec, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_ec.started')
            # update status
            text_ec.status = STARTED
            text_ec.setAutoDraw(True)
        
        # if text_ec is active this frame...
        if text_ec.status == STARTED:
            # update params
            pass
        
        # if text_ec is stopping this frame...
        if text_ec.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text_ec.tStartRefresh + 180.0-frameTolerance:
                # keep track of stop time/frame for later
                text_ec.tStop = t  # not accounting for scr refresh
                text_ec.tStopRefresh = tThisFlipGlobal  # on global time
                text_ec.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_ec.stopped')
                # update status
                text_ec.status = FINISHED
                text_ec.setAutoDraw(False)
        
        # if tone_finish is starting this frame...
        if tone_finish.status == NOT_STARTED and tThisFlip >= 180.0-frameTolerance:
            # keep track of start time/frame for later
            tone_finish.frameNStart = frameN  # exact frame index
            tone_finish.tStart = t  # local t and not account for scr refresh
            tone_finish.tStartRefresh = tThisFlipGlobal  # on global time
            # add timestamp to datafile
            thisExp.addData('tone_finish.started', tThisFlipGlobal)
            # update status
            tone_finish.status = STARTED
            tone_finish.play(when=win)  # sync with win flip
        
        # if tone_finish is stopping this frame...
        if tone_finish.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > tone_finish.tStartRefresh + 0.3-frameTolerance:
                # keep track of stop time/frame for later
                tone_finish.tStop = t  # not accounting for scr refresh
                tone_finish.tStopRefresh = tThisFlipGlobal  # on global time
                tone_finish.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'tone_finish.stopped')
                # update status
                tone_finish.status = FINISHED
                tone_finish.stop()
        
        # if tone_finish is active this frame...
        if tone_finish.status == STARTED:
            # update params
            pass
            # if sound is finished but tone_finish has time left, finish it now
            if tone_finish.isFinished:
                tone_finish.status = FINISHED
        # Run 'Each Frame' code from trigger_ec
        if text_ec.status == STARTED and not stimulus_pulse_started:
            win.callOnFlip(dev.activate_line, bitmask=block_start_code)
            win.callOnFlip(eyetracker.sendMessage, block_start_code)
            stimulus_pulse_started = True
        
        if text_ec.status == FINISHED and not stimulus_pulse_started_2:
            dev.activate_line(bitmask=block_end_code)
            eyetracker.sendMessage(block_end_code)
            stimulus_pulse_started_2 = True
        
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in eyes_closedComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "eyes_closed" ---
    for thisComponent in eyes_closedComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('eyes_closed.stopped', globalClock.getTime(format='float'))
    tone_finish.pause()  # ensure sound has stopped at end of Routine
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-180.300000)
    thisExp.nextEntry()
    
    # --- Prepare to start Routine "instruct_eo" ---
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_instruct_2
    key_instruct_2.keys = []
    key_instruct_2.rt = []
    _key_instruct_2_allKeys = []
    read_instruct_2.setSound('resource/instruct_eo_reading_audio.wav', hamming=True)
    read_instruct_2.setVolume(1.0, log=False)
    read_instruct_2.seek(0)
    # keep track of which components have finished
    instruct_eoComponents = [text_instruct_2, key_instruct_2, read_instruct_2]
    for thisComponent in instruct_eoComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "instruct_eo" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_instruct_2* updates
        
        # if text_instruct_2 is starting this frame...
        if text_instruct_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_instruct_2.frameNStart = frameN  # exact frame index
            text_instruct_2.tStart = t  # local t and not account for scr refresh
            text_instruct_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_instruct_2, 'tStartRefresh')  # time at next scr refresh
            # update status
            text_instruct_2.status = STARTED
            text_instruct_2.setAutoDraw(True)
        
        # if text_instruct_2 is active this frame...
        if text_instruct_2.status == STARTED:
            # update params
            pass
        
        # *key_instruct_2* updates
        waitOnFlip = False
        
        # if key_instruct_2 is starting this frame...
        if key_instruct_2.status == NOT_STARTED and tThisFlip >= 0.2-frameTolerance:
            # keep track of start time/frame for later
            key_instruct_2.frameNStart = frameN  # exact frame index
            key_instruct_2.tStart = t  # local t and not account for scr refresh
            key_instruct_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_instruct_2, 'tStartRefresh')  # time at next scr refresh
            # update status
            key_instruct_2.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_instruct_2.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_instruct_2.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_instruct_2.status == STARTED and not waitOnFlip:
            theseKeys = key_instruct_2.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=True)
            _key_instruct_2_allKeys.extend(theseKeys)
            if len(_key_instruct_2_allKeys):
                key_instruct_2.keys = _key_instruct_2_allKeys[0].name  # just the first key pressed
                key_instruct_2.rt = _key_instruct_2_allKeys[0].rt
                key_instruct_2.duration = _key_instruct_2_allKeys[0].duration
                # a response ends the routine
                continueRoutine = False
        
        # if read_instruct_2 is starting this frame...
        if read_instruct_2.status == NOT_STARTED and tThisFlip >= 0.8-frameTolerance:
            # keep track of start time/frame for later
            read_instruct_2.frameNStart = frameN  # exact frame index
            read_instruct_2.tStart = t  # local t and not account for scr refresh
            read_instruct_2.tStartRefresh = tThisFlipGlobal  # on global time
            # update status
            read_instruct_2.status = STARTED
            read_instruct_2.play(when=win)  # sync with win flip
        
        # if read_instruct_2 is active this frame...
        if read_instruct_2.status == STARTED:
            # update params
            pass
            # if sound is finished but read_instruct_2 has time left, finish it now
            if read_instruct_2.isFinished:
                read_instruct_2.status = FINISHED
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in instruct_eoComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instruct_eo" ---
    for thisComponent in instruct_eoComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # check responses
    if key_instruct_2.keys in ['', [], None]:  # No response was made
        key_instruct_2.keys = None
    thisExp.addData('key_instruct_2.keys',key_instruct_2.keys)
    if key_instruct_2.keys != None:  # we had a response
        thisExp.addData('key_instruct_2.rt', key_instruct_2.rt)
        thisExp.addData('key_instruct_2.duration', key_instruct_2.duration)
    read_instruct_2.pause()  # ensure sound has stopped at end of Routine
    thisExp.nextEntry()
    # the Routine "instruct_eo" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "eyes_open" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('eyes_open.started', globalClock.getTime(format='float'))
    # Run 'Begin Routine' code from trigger_eo
    stimulus_pulse_started = False
    stimulus_pulse_started_2 = False
    
    # keep track of which components have finished
    eyes_openComponents = [text_eo]
    for thisComponent in eyes_openComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "eyes_open" ---
    routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 180.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_eo* updates
        
        # if text_eo is starting this frame...
        if text_eo.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_eo.frameNStart = frameN  # exact frame index
            text_eo.tStart = t  # local t and not account for scr refresh
            text_eo.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_eo, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_eo.started')
            # update status
            text_eo.status = STARTED
            text_eo.setAutoDraw(True)
        
        # if text_eo is active this frame...
        if text_eo.status == STARTED:
            # update params
            pass
        
        # if text_eo is stopping this frame...
        if text_eo.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text_eo.tStartRefresh + 180.0-frameTolerance:
                # keep track of stop time/frame for later
                text_eo.tStop = t  # not accounting for scr refresh
                text_eo.tStopRefresh = tThisFlipGlobal  # on global time
                text_eo.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_eo.stopped')
                # update status
                text_eo.status = FINISHED
                text_eo.setAutoDraw(False)
        # Run 'Each Frame' code from trigger_eo
        if text_eo.status == STARTED and not stimulus_pulse_started:
            win.callOnFlip(dev.activate_line, bitmask=block_start_code)
            win.callOnFlip(eyetracker.sendMessage, block_start_code)
            stimulus_pulse_started = True
        
        if text_eo.status == FINISHED and not stimulus_pulse_started_2:
            dev.activate_line(bitmask=block_end_code)
            eyetracker.sendMessage(block_end_code)
            stimulus_pulse_started_2 = True
        
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in eyes_openComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "eyes_open" ---
    for thisComponent in eyes_openComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('eyes_open.stopped', globalClock.getTime(format='float'))
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-180.000000)
    thisExp.nextEntry()
    
    # --- Prepare to start Routine "_thank_you" ---
    continueRoutine = True
    # update component parameters for each repeat
    read_thank_you.setSound('resource/thank_you.wav', secs=2.7, hamming=True)
    read_thank_you.setVolume(1.0, log=False)
    read_thank_you.seek(0)
    # keep track of which components have finished
    _thank_youComponents = [text_thank_you, read_thank_you]
    for thisComponent in _thank_youComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "_thank_you" ---
    routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 3.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_thank_you* updates
        
        # if text_thank_you is starting this frame...
        if text_thank_you.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_thank_you.frameNStart = frameN  # exact frame index
            text_thank_you.tStart = t  # local t and not account for scr refresh
            text_thank_you.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_thank_you, 'tStartRefresh')  # time at next scr refresh
            # update status
            text_thank_you.status = STARTED
            text_thank_you.setAutoDraw(True)
        
        # if text_thank_you is active this frame...
        if text_thank_you.status == STARTED:
            # update params
            pass
        
        # if text_thank_you is stopping this frame...
        if text_thank_you.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text_thank_you.tStartRefresh + 3.0-frameTolerance:
                # keep track of stop time/frame for later
                text_thank_you.tStop = t  # not accounting for scr refresh
                text_thank_you.tStopRefresh = tThisFlipGlobal  # on global time
                text_thank_you.frameNStop = frameN  # exact frame index
                # update status
                text_thank_you.status = FINISHED
                text_thank_you.setAutoDraw(False)
        
        # if read_thank_you is starting this frame...
        if read_thank_you.status == NOT_STARTED and tThisFlip >= 0.2-frameTolerance:
            # keep track of start time/frame for later
            read_thank_you.frameNStart = frameN  # exact frame index
            read_thank_you.tStart = t  # local t and not account for scr refresh
            read_thank_you.tStartRefresh = tThisFlipGlobal  # on global time
            # update status
            read_thank_you.status = STARTED
            read_thank_you.play(when=win)  # sync with win flip
        
        # if read_thank_you is stopping this frame...
        if read_thank_you.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > read_thank_you.tStartRefresh + 2.7-frameTolerance:
                # keep track of stop time/frame for later
                read_thank_you.tStop = t  # not accounting for scr refresh
                read_thank_you.tStopRefresh = tThisFlipGlobal  # on global time
                read_thank_you.frameNStop = frameN  # exact frame index
                # update status
                read_thank_you.status = FINISHED
                read_thank_you.stop()
        
        # if read_thank_you is active this frame...
        if read_thank_you.status == STARTED:
            # update params
            pass
            # if sound is finished but read_thank_you has time left, finish it now
            if read_thank_you.isFinished:
                read_thank_you.status = FINISHED
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in _thank_youComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "_thank_you" ---
    for thisComponent in _thank_youComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    read_thank_you.pause()  # ensure sound has stopped at end of Routine
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-3.000000)
    thisExp.nextEntry()
    # Run 'End Experiment' code from eeg
    # Stop EEG recording
    dev.activate_line(bitmask=127)  # trigger 127 will stop EEG
    
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # return console logger level to WARNING
    logging.console.setLevel(logging.WARNING)
    # mark experiment handler as finished
    thisExp.status = FINISHED
    # shut down eyetracker, if there is one
    if deviceManager.getDevice('eyetracker') is not None:
        deviceManager.removeDevice('eyetracker')
    logging.flush()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    # shut down eyetracker, if there is one
    if deviceManager.getDevice('eyetracker') is not None:
        deviceManager.removeDevice('eyetracker')
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)
