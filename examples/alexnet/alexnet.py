#!/usr/local/bin/pyleabra -i

# Copyright (c) 2020, The Emergent Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# AlexNet network copied directly from pytorch torchvision github repo:
# https://github.com/pytorch/vision/blob/master/torchvision/models/alexnet.py

from leabra import go, etorch, etor, emer, relpos, eplot, env, agg, patgen, prjn, etable, efile, split, etensor, params, netview, rand, erand, gi, giv, pygiv, pyparams, pyet, mat32

import io, sys, getopt
from datetime import datetime, timezone
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.model_zoo import load_url as load_state_dict_from_url
from typing import Any

# sample execution (requires torchvision)
from PIL import Image
from torchvision import transforms

# import matplotlib
# matplotlib.use('SVG')
# import matplotlib.pyplot as plt
# plt.rcParams['svg.fonttype'] = 'none'  # essential for not rendering fonts as paths

# this will become Sim later.. 
TheSim = 1

# LogPrec is precision for saving float values in logs
LogPrec = 4

model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}

images = ['images/dog1.jpg', 'images/dog2.jpg', 'images/fish1.jpg', 'images/fish2.jpg']

class AlexNet(nn.Module):

    def __init__(self, num_classes: int = 1000) -> None:
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),  # C1.Net
            nn.ReLU(inplace=True), # C1.Act
            nn.MaxPool2d(kernel_size=3, stride=2), # P1.Act
            nn.Conv2d(64, 192, kernel_size=5, padding=2), # C2.Net
            nn.ReLU(inplace=True), # C2.Act
            nn.MaxPool2d(kernel_size=3, stride=2), # P2.Act
            nn.Conv2d(192, 384, kernel_size=3, padding=1),  # C3.Net
            nn.ReLU(inplace=True),  # C3.Act
            nn.Conv2d(384, 256, kernel_size=3, padding=1), # C4.Net
            nn.ReLU(inplace=True),  # C4.Act
            nn.Conv2d(256, 256, kernel_size=3, padding=1), # C5.Net
            nn.ReLU(inplace=True), # C5.Act
            nn.MaxPool2d(kernel_size=3, stride=2),  # P5.Act
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),  # D1 -- ignore
            nn.Linear(256 * 6 * 6, 4096), # Cl1.Net
            nn.ReLU(inplace=True), # Cl1.Act
            nn.Dropout(),  # D2 
            nn.Linear(4096, 4096), # Cl2.Net
            nn.ReLU(inplace=True),  # Cl2.Act
            nn.Linear(4096, num_classes), # Out.Net
        )
        self.est = etor.State(self)
        self.fnames = ["C1.Net", "C1.Act", "P1.Act", "C2.Net", "C2.Act", "P2.Act", "C3.Net", "C3.Act", "C4.Net", "C4.Act", "C5.Net", "C5.Act", "P5.Act"]
        self.cnames = ["", "Cl1.Net", "Cl1.Act", "", "Cl2.Net", "Cl2.Act", "Out.Net"]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.est.rec(x, "Image.Act")
        for i, f in enumerate(self.features):
            x = f(x)
            self.est.rec(x, self.fnames[i])
        x = self.avgpool(x)
        self.est.rec(x, "AP.Act")
        x = torch.flatten(x, 1)
        for i, f in enumerate(self.classifier):
            x = f(x)
            cnm = self.cnames[i]
            if len(cnm) > 0:
                self.est.rec(x, cnm)
        x = torch.nn.functional.softmax(x[0], dim=0)
        self.est.rec(x, "Out.Act")
        return x

def alexnet(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> AlexNet:
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = AlexNet(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['alexnet'],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


# note: we cannot use methods for callbacks from Go -- must be separate functions
# so below are all the callbacks from the GUI toolbar actions

def InitCB(recv, send, sig, data):
    TheSim.Init()
    TheSim.UpdateClassView()
    TheSim.vp.SetNeedsFullRender()

def StopCB(recv, send, sig, data):
    TheSim.Stop()

def TestTrialCB(recv, send, sig, data):
    if not TheSim.IsRunning:
        TheSim.IsRunning = True
        TheSim.TestTrial(False)
        TheSim.IsRunning = False
        TheSim.UpdateClassView()
        TheSim.vp.SetNeedsFullRender()

def TestItemCB2(recv, send, sig, data):
    win = gi.Window(handle=recv)
    vp = win.WinViewport2D()
    dlg = gi.Dialog(handle=send)
    if sig != gi.DialogAccepted:
        return
    val = gi.StringPromptDialogValue(dlg)
    idx = -1
    for i, img in enumerate(images):
        if val in img:
            idx = i
            break
    if idx < 0:
        gi.PromptDialog(vp, gi.DlgOpts(Title="Name Not Found", Prompt="No patterns found containing: " + val), True, False, go.nil, go.nil)
    else:
        if not TheSim.IsRunning:
            TheSim.IsRunning = True
            print("testing index: %s" % idx)
            TheSim.TestItem(idx)
            TheSim.IsRunning = False
            vp.SetNeedsFullRender()

def TestItemCB(recv, send, sig, data):
    win = gi.Window(handle=recv)
    gi.StringPromptDialog(win.WinViewport2D(), "", "Test Item",
        gi.DlgOpts(Title="Test Item", Prompt="Enter the Name of a given input pattern to test (case insensitive, contains given string."), win, TestItemCB2)

def TestAllCB(recv, send, sig, data):
    if not TheSim.IsRunning:
        TheSim.IsRunning = True
        TheSim.ToolBar.UpdateActions()
        TheSim.RunTestAll()

def ResetRunLogCB(recv, send, sig, data):
    TheSim.RunLog.SetNumRows(0)
    TheSim.RunPlot.Update()

def NewRndSeedCB(recv, send, sig, data):
    TheSim.NewRndSeed()

def ReadmeCB(recv, send, sig, data):
    gi.OpenURL("https://github.com/emer/etorch/blob/master/examples/alexnet/README.md")

def FilterErr(et, row):
    return etable.Table(handle=et).CellFloat("Err", row) > 0 # include error trials    

def UpdtFuncNotRunning(act):
    act.SetActiveStateUpdt(not TheSim.IsRunning)
    
def UpdtFuncRunning(act):
    act.SetActiveStateUpdt(TheSim.IsRunning)

    
#####################################################    
#     Sim

class Sim(pygiv.ClassViewObj):
    """
    Sim encapsulates the entire simulation model, and we define all the
    functionality as methods on this struct.  This structure keeps all relevant
    state information organized and available without having to pass everything around
    as arguments to methods, and provides the core GUI interface (note the view tags
    for the fields which provide hints to how things should be displayed).
    """

    def __init__(self):
        super(Sim, self).__init__()
        self.TorchNet = 0
        self.SetTags("TorchNet", 'view:"-" desc:"the Torch network"')
        self.Optimizer = 0 
        self.SetTags("Optimizer", 'view:"-" desc:"the Torch optimizer"')
        self.Criterion = 0 
        self.SetTags("Criterion", 'view:"-" desc:"the Torch criterion"')
        self.CurInput = 0
        self.SetTags("CurInput", 'view:"-" desc:"the current input tensor"')
        self.idx2label = []
        self.SetTags("idx2label", 'view:"-" desc:"output index to label mapping"')
        self.cls2label = {}
        self.SetTags("cls2label", 'view:"-" desc:"output class to label mapping"')
        
        self.Net = etorch.Network()
        self.SetTags("Net", 'view:"no-inline" desc:"the network -- click to view / edit parameters for layers, prjns, etc"')

        self.TstEpcLog = etable.Table()
        self.SetTags("TstEpcLog", 'view:"no-inline" desc:"testing epoch-level log data"')
        self.TstTrlLog = etable.Table()
        self.SetTags("TstTrlLog", 'view:"no-inline" desc:"testing trial-level log data"')
        self.TstErrLog = etable.Table()
        self.SetTags("TstErrLog", 'view:"no-inline" desc:"log of all test trials where errors were made"')
        self.TstErrStats = etable.Table()
        self.SetTags("TstErrStats", 'view:"no-inline" desc:"stats on test trials where errors were made"')
        self.Params = params.Sets()
        self.SetTags("Params", 'view:"no-inline" desc:"full collection of param sets"')
        self.ViewOn = True
        self.SetTags("ViewOn", 'desc:"whether to update the network view while running"')
        self.ViewWts = False
        self.SetTags("ViewWts", 'desc:"whether to update the network weights view while running -- slower than acts"')

        # statistics: note use float64 as that is best for etable.Table
        self.TestIdx = int(-1)
        self.SetTags("TestIdx", 'inactive:"+" desc:"index of item to test"')
        self.TestImg = str()
        self.SetTags("TestImg", 'inactive:"+" desc:"image name to test"')
        self.Top5 = []
        self.SetTags("Top5", 'width:"80" inactive:"+" desc:"top 5 output labels, in order"')
        self.Top5Vals = []
        self.SetTags("Top5Vals", 'inactive:"+" desc:"top 5 output softmax values in order"')
        self.TrlErr = float()
        self.SetTags("TrlErr", 'inactive:"+" desc:"1 if trial was error, 0 if correct -- based on SSE = 0 (subject to .5 unit-wise tolerance)"')
        self.EpcPctErr = float()
        self.SetTags("EpcPctErr", 'inactive:"+" desc:"last epoch\'s average TrlErr"')
        self.EpcPctCor = float()
        self.SetTags("EpcPctCor", 'inactive:"+" desc:"1 - last epoch\'s average TrlErr"')
        self.EpcPerTrlMSec = float()
        self.SetTags("EpcPerTrlMSec", 'inactive:"+" desc:"how long did the epoch take per trial in wall-clock milliseconds"')

        # internal state - view:"-"
        self.SumErr = float()
        self.SetTags("SumErr", 'view:"-" inactive:"+" desc:"sum to increment as we go through epoch"')

        self.Win = 0
        self.SetTags("Win", 'view:"-" desc:"main GUI window"')
        self.NetView = 0
        self.SetTags("NetView", 'view:"-" desc:"the network viewer"')
        self.ToolBar = 0
        self.SetTags("ToolBar", 'view:"-" desc:"the master toolbar"')
        self.TstEpcPlot = 0
        self.SetTags("TstEpcPlot", 'view:"-" desc:"the testing epoch plot"')
        self.TstTrlPlot = 0
        self.SetTags("TstTrlPlot", 'view:"-" desc:"the test-trial plot"')
        self.ValsTsrs = {}
        self.SetTags("ValsTsrs", 'view:"-" desc:"for holding layer values"')
        self.NoGui = False
        self.SetTags("NoGui", 'view:"-" desc:"if true, runing in no GUI mode"')
        self.IsRunning = False
        self.SetTags("IsRunning", 'view:"-" desc:"true if sim is running"')
        self.StopNow = False
        self.SetTags("StopNow", 'view:"-" desc:"flag to stop running"')
        self.RndSeed = int(1)
        self.SetTags("RndSeed", 'view:"-" desc:"the current random seed"')
        self.LastEpcTime = int()
        self.SetTags("LastEpcTime", 'view:"-" desc:"timer for last epoch"')
        self.vp  = 0 
        self.SetTags("vp", 'view:"-" desc:"viewport"')

    def InitParams(ss):
        """
        Sets the default set of parameters -- Base is always applied, and others can be optionally
        selected to apply on top of that
        """
        # ss.Params.OpenJSON("ra25_std.params")

    def Config(ss):
        """
        Config configures all the elements using the standard functions
        """
        ss.InitParams()
        ss.ConfigEnv()
        ss.ConfigNet(ss.Net)
        ss.ConfigTstEpcLog(ss.TstEpcLog)
        ss.ConfigTstTrlLog(ss.TstTrlLog)

    def ConfigEnv(ss):
        with open("imagenet_class_index.json", "r") as read_file:
            class_idx = json.load(read_file)
            ss.idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]
            ss.cls2label = {class_idx[str(k)][0]: class_idx[str(k)][1] for k in range(len(class_idx))}

    def ConfigNet(ss, net):
        ss.TorchNet = alexnet(pretrained = True)
        ss.TorchNet.est.trace = False  # turn this on to see size of state vars
        ss.TorchNet.est.rec_wts = False
        ss.TorchNet.eval()  # evaluation, not training mode -- no dropout

        print(ss.TorchNet)
        for pt in ss.TorchNet.state_dict():
            print(pt, "\t", ss.TorchNet.state_dict()[pt].size())

        net.InitName(net, "AlexNet")
        img = net.AddLayer4D("Image", 1, 3, 224, 224, emer.Input)
        c1 = net.AddLayer4D("C1", 8, 8, 55, 55, emer.Hidden)   # note: feature is outer, not inner here
        p1 = net.AddLayer4D("P1", 8, 8, 27, 27, emer.Hidden)
        c2 = net.AddLayer4D("C2", 12, 16, 27, 27, emer.Hidden)
        p2 = net.AddLayer4D("P2", 12, 16, 13, 13, emer.Hidden)
        c3 = net.AddLayer4D("C3", 24, 16, 13, 13, emer.Hidden)
        c4 = net.AddLayer4D("C4", 16, 16, 13, 13, emer.Hidden)
        c5 = net.AddLayer4D("C5", 16, 16, 13, 13, emer.Hidden)
        p5 = net.AddLayer4D("P5", 16, 16, 6, 6, emer.Hidden)
        ap = net.AddLayer4D("AP", 16, 16, 6, 6, emer.Hidden)
        cl1 = net.AddLayer2D("Cl1", 64, 64, emer.Hidden)
        cl2 = net.AddLayer2D("Cl2", 64, 64, emer.Hidden)
        out = net.AddLayer2D("Out", 40, 25, emer.Target)
        
        c1.SetRelPos(relpos.Rel(Rel=relpos.Above, Other="Image", XAlign=relpos.Left, YAlign=relpos.Front))
        p1.SetRelPos(relpos.Rel(Rel=relpos.RightOf, Other="C1", YAlign=relpos.Front, Space=10))
        c2.SetRelPos(relpos.Rel(Rel=relpos.Above, Other="C1", XAlign=relpos.Left, YAlign=relpos.Front))
        p2.SetRelPos(relpos.Rel(Rel=relpos.RightOf, Other="C2", YAlign=relpos.Front, Space=10))
        c3.SetRelPos(relpos.Rel(Rel=relpos.Above, Other="C2", XAlign=relpos.Left, YAlign=relpos.Front))
        c4.SetRelPos(relpos.Rel(Rel=relpos.RightOf, Other="C3", XAlign=relpos.Front, Space=10))
        c5.SetRelPos(relpos.Rel(Rel=relpos.RightOf, Other="C4", YAlign=relpos.Front, Space=10))
        p5.SetRelPos(relpos.Rel(Rel=relpos.Above, Other="C3", XAlign=relpos.Left, YAlign=relpos.Front))
        ap.SetRelPos(relpos.Rel(Rel=relpos.RightOf, Other="P5", YAlign=relpos.Front, Space=10))
        cl1.SetRelPos(relpos.Rel(Rel=relpos.RightOf, Other="AP", YAlign=relpos.Front, Space=10))
        cl2.SetRelPos(relpos.Rel(Rel=relpos.RightOf, Other="Cl1", YAlign=relpos.Front, Space=10))
        out.SetRelPos(relpos.Rel(Rel=relpos.RightOf, Other="Cl2", YAlign=relpos.Front, Space=10))

        net.Defaults()
        ss.SetParams("Network", False) # only set Network params
        net.Build()
        
        ss.TorchNet.est.init_net(net)  # grabs all the info from network

    def Init(ss):
        """
        Init restarts the run, and initializes everything, including network weights
        and resets the epoch log table
        """
        rand.Seed(ss.RndSeed)
        ss.ConfigEnv()

        ss.StopNow = False
        ss.SetParams("", False) # all sheets
        ss.TorchNet.est.rec_wts = ss.ViewWts
        ss.UpdateView(False)

    def NewRndSeed(ss):
        """
        NewRndSeed gets a new random seed based on current time -- otherwise uses
        the same random seed for every run
        """
        ss.RndSeed = int(datetime.now(timezone.utc).timestamp())

    def Counters(ss, train):
        """
        Counters returns a string of the current counter state
        use tabs to achieve a reasonable formatting overall
        and add a few tabs at the end to allow for expansion..
        """
        return "Trial:\t%d\tName:\t%s\t\t\t" % (ss.TestIdx, ss.TestImg)

    def UpdateView(ss, train):
        if ss.NetView != 0 and ss.NetView.IsVisible():
            ss.TorchNet.est.update()  # does everything
            ss.NetView.Record(ss.Counters(train))
            ss.NetView.GoUpdate()

    def ApplyInputs(ss, img):
        """
        ApplyInputs just grabs current pattern into torch vectors
        """
        input_image = Image.open(img)
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = preprocess(input_image)
        ss.CurInput = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

    def TorchTrial(ss, train):
        """
        Does one trial of Torch network training.
        """
        if ss.Win != 0:
            ss.Win.PollEvents() # this is essential for GUI responsiveness while running

        if train:
            ss.Optimizer.zero_grad()
        out = ss.TorchNet(ss.CurInput)
        pred = torch.topk(out, 5)
        ss.Top5 = []
        ss.Top5Vals = pred.values
        for i in pred.indices:
            lab = ss.idx2label[i]
            ss.Top5.append(lab)
        
        print(ss.Top5, ss.Top5Vals)
        # loss = ss.Criterion(out.squeeze(), ss.TOutPatsTrl)
        # ss.TrlSSE = loss.item()
        if train:
            loss.backward()
            ss.Optimizer.step()
        ss.UpdateView(train)

    def InitStats(ss):
        """
        InitStats initializes all the statistics, especially important for the
        cumulative epoch stats -- called at start of new run
        """
        ss.SumErr = 0
        ss.TrlErr = 0
        ss.EpcPctErr = 0

    def TrialStats(ss, accum):
        """
        TrialStats computes the trial-level statistics and adds them to the epoch accumulators if
        accum is true.  Note that we're accumulating stats here on the Sim side so the
        core algorithm side remains as simple as possible, and doesn't need to worry about
        different time-scales over which stats could be accumulated etc.
        You can also aggregate directly from log data, as is done for testing stats
        """
        ss.TrlErr = 0 # todo -- don't know labels..
        if accum:
            ss.SumErr += ss.TrlErr

    def Stop(ss):
        """
        Stop tells the sim to stop running
        """
        ss.StopNow = True

    def Stopped(ss):
        """
        Stopped is called when a run method stops running -- updates the IsRunning flag and toolbar
        """
        ss.IsRunning = False
        if ss.Win != 0:
            vp = ss.Win.WinViewport2D()
            if ss.ToolBar != 0:
                ss.ToolBar.UpdateActions()
            vp.SetNeedsFullRender()
            ss.UpdateClassView()

    def TestTrial(ss, returnOnChg):
        """
        TestTrial runs one trial of testing -- always sequentially presented inputs
        """
        ss.TestIdx += 1
        if ss.TestIdx >= len(images):
            ss.TestIdx = 0
        ss.TestImg = images[ss.TestIdx]
        ss.ApplyInputs(ss.TestImg)
        ss.TorchTrial(False)
        ss.TrialStats(False)
        ss.LogTstTrl(ss.TstTrlLog)

    def TestItem(ss, idx):
        """
        TestItem tests given item which is at given index in test item list
        """
        ss.TestIdx = idx
        ss.TestImg = images[ss.TestIdx]
        ss.ApplyInputs(ss.TestImg)
        ss.TorchTrial(False)
        ss.TrialStats(False)

    def TestAll(ss):
        """
        TestAll runs through the full set of testing items
        """
        ss.TestIdx = -1
        for i in range(len(images)):
            ss.TestTrial(True)

    def RunTestAll(ss):
        """
        RunTestAll runs through the full set of testing items, has stop running = false at end -- for gui
        """
        ss.StopNow = False
        ss.TestAll()
        ss.Stopped()

    def ParamsName(ss):
        """
        ParamsName returns name of current set of parameters
        """
        if ss.ParamSet == "":
            return "Base"
        return ss.ParamSet

    def SetParams(ss, sheet, setMsg):
        """
        SetParams sets the params for "Base" and then current ParamSet.
        If sheet is empty, then it applies all avail sheets (e.g., Network, Sim)
        otherwise just the named sheet
        if setMsg = true then we output a message for each param that was set.
        """
        
        # for param_group in ss.Optimizer.param_groups:
        #     param_group['lr'] = ss.Lrate

        # todo: fancy params stuff not yet supported...
        return
        
        if sheet == "":
            ss.Params.ValidateSheets(go.Slice_string(["Network", "Sim"]))
        ss.SetParamsSet("Base", sheet, setMsg)
        if ss.ParamSet != "" and ss.ParamSet != "Base":
            sps = ss.ParamSet.split()
            for ps in sps:
                ss.SetParamsSet(ps, sheet, setMsg)

    def SetParamsSet(ss, setNm, sheet, setMsg):
        """
        SetParamsSet sets the params for given params.Set name.
        If sheet is empty, then it applies all avail sheets (e.g., Network, Sim)
        otherwise just the named sheet
        if setMsg = true then we output a message for each param that was set.
        """
        pset = ss.Params.SetByNameTry(setNm)
        if sheet == "" or sheet == "Network":
            if "Network" in pset.Sheets:
                netp = pset.SheetByNameTry("Network")
                ss.Net.ApplyParams(netp, setMsg)

        if sheet == "" or sheet == "Sim":
            if "Sim" in pset.Sheets:
                simp= pset.SheetByNameTry("Sim")
                pyparams.ApplyParams(ss, simp, setMsg)

    def ValsTsr(ss, name):
        """
        ValsTsr gets value tensor of given name, creating if not yet made
        """
        if name in ss.ValsTsrs:
            return ss.ValsTsrs[name]
        tsr = etensor.Float32()
        ss.ValsTsrs[name] = tsr
        return tsr

    def RunName(ss):
        """
        RunName returns a name for this run that combines Tag and Params -- add this to
        any file names that are saved.
        """
        if ss.Tag != "":
            return ss.Tag + "_" + ss.ParamsName()
        else:
            return ss.ParamsName()

    def LogFileName(ss, lognm):
        """
        LogFileName returns default log file name
        """
        return ss.Net.Nm + "_" + ss.RunName() + "_" + lognm + ".tsv"

    def LogTstTrl(ss, dt):
        """
        LogTstTrl adds data from current trial to the TstTrlLog table.
        log always contains number of testing items
        """
        epc = 0
        inp = etorch.Layer(ss.Net.LayerByName("Image"))
        out = etorch.Layer(ss.Net.LayerByName("Out"))

        trl = ss.TestIdx
        row = trl

        if dt.Rows <= row:
            dt.SetNumRows(row + 1)

        dt.SetCellFloat("Run", row, float(0))
        dt.SetCellFloat("Epoch", row, float(epc))
        dt.SetCellFloat("Trial", row, float(trl))
        dt.SetCellString("TrialName", row, ss.TestImg)
        dt.SetCellString("Top5", row, str(ss.Top5))
        for i, v in enumerate(ss.Top5Vals):
            dt.SetCellTensorFloat1D("Top5Vals", row, i, v)
        dt.SetCellFloat("Err", row, ss.TrlErr)

        ivt = ss.ValsTsr("Image")
        ovt = ss.ValsTsr("Out")
        inp.UnitValsTensor(ivt, "Act")
        dt.SetCellTensor("InAct", row, ivt)
        out.UnitValsTensor(ovt, "Act")
        dt.SetCellTensor("OutAct", row, ovt)

        if ss.TstTrlPlot != 0:
            ss.TstTrlPlot.GoUpdate()

    def ConfigTstTrlLog(ss, dt):
        inp = etorch.Layer(ss.Net.LayerByName("Image"))
        out = etorch.Layer(ss.Net.LayerByName("Out"))

        dt.SetMetaData("name", "TstTrlLog")
        dt.SetMetaData("desc", "Record of testing per input pattern")
        dt.SetMetaData("read-only", "true")
        dt.SetMetaData("precision", str(LogPrec))

        nt = len(images)
        sch = etable.Schema(
            [etable.Column("Run", etensor.INT64, go.nil, go.nil),
            etable.Column("Epoch", etensor.INT64, go.nil, go.nil),
            etable.Column("Trial", etensor.INT64, go.nil, go.nil),
            etable.Column("TrialName", etensor.STRING, go.nil, go.nil),
            etable.Column("Top5", etensor.STRING, go.nil, go.nil),
            etable.Column("Top5Vals", etensor.FLOAT64, go.Slice_int([5]), go.nil),
            etable.Column("Err", etensor.FLOAT64, go.nil, go.nil)]
        )
        sch.append(etable.Column("InAct", etensor.FLOAT64, inp.Shp.Shp, go.nil))
        sch.append(etable.Column("OutAct", etensor.FLOAT64, out.Shp.Shp, go.nil))
        dt.SetFromSchema(sch, nt)

    def ConfigTstTrlPlot(ss, plt, dt):
        plt.Params.Title = "eTorch AlexNet Test Trial Plot"
        plt.Params.XAxisCol = "Trial"
        plt.SetTable(dt)
        # order of params: on, fixMin, min, fixMax, max
        plt.SetColParams("Run", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("Epoch", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("Trial", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("TrialName", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("Top5", eplot.On, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("Top5Vals", eplot.On, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("Err", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)

        plt.SetColParams("InAct", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
        plt.SetColParams("OutAct", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
        return plt

    def LogTstEpc(ss, dt):
        row = dt.Rows
        dt.SetNumRows(row + 1)

        trl = ss.TstTrlLog
        tix = etable.NewIdxView(trl)
        epc = 0

        # note: this shows how to use agg methods to compute summary data from another
        # data table, instead of incrementing on the Sim
        dt.SetCellFloat("Run", row, float(0))
        dt.SetCellFloat("Epoch", row, float(epc))
        dt.SetCellFloat("PctErr", row, agg.Mean(tix, "Err")[0])
        dt.SetCellFloat("PctCor", row, 1-agg.Mean(tix, "Err")[0])

        trlix = etable.NewIdxView(trl)
        trlix.Filter(FilterErr) # requires separate function

        ss.TstErrLog = trlix.NewTable()

        allsp = split.All(trlix)
        split.Agg(allsp, "InAct", agg.AggMean)
        split.Agg(allsp, "OutAct", agg.AggMean)

        ss.TstErrStats = allsp.AggsToTable(etable.AddAggName)

        # note: essential to use Go version of update when called from another goroutine
        if ss.TstEpcPlot != 0:
            ss.TstEpcPlot.GoUpdate()

    def ConfigTstEpcLog(ss, dt):
        dt.SetMetaData("name", "TstEpcLog")
        dt.SetMetaData("desc", "Summary stats for testing trials")
        dt.SetMetaData("read-only", "true")
        dt.SetMetaData("precision", str(LogPrec))

        sch = etable.Schema(
            [etable.Column("Run", etensor.INT64, go.nil, go.nil),
            etable.Column("Epoch", etensor.INT64, go.nil, go.nil),
            etable.Column("PctErr", etensor.FLOAT64, go.nil, go.nil),
            etable.Column("PctCor", etensor.FLOAT64, go.nil, go.nil)]
        )
        dt.SetFromSchema(sch, 0)

    def ConfigTstEpcPlot(ss, plt, dt):
        plt.Params.Title = "eTorch AlexNet Testing Epoch Plot"
        plt.Params.XAxisCol = "Epoch"
        plt.SetTable(dt)
        # order of params: on, fixMin, min, fixMax, max
        plt.SetColParams("Run", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("Epoch", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("PctErr", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1) # default plot
        plt.SetColParams("PctCor", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1) # default plot
        return plt

    def ConfigGui(ss):
        """
        ConfigGui configures the GoGi gui interface for this simulation,
        """
        width = 1600
        height = 1200

        gi.SetAppName("alexnet")
        gi.SetAppAbout('This tests AlexNet so you can visualize its response to inputs. See <a href="https://github.com/emer/emergent">emergent on GitHub</a>.</p>')

        win = gi.NewMainWindow("AlexNet", "eTorch AlexNet", width, height)
        ss.Win = win

        vp = win.WinViewport2D()
        ss.vp = vp
        updt = vp.UpdateStart()

        mfr = win.SetMainFrame()

        tbar = gi.AddNewToolBar(mfr, "tbar")
        tbar.SetStretchMaxWidth()
        ss.ToolBar = tbar

        split = gi.AddNewSplitView(mfr, "split")
        split.Dim = mat32.X
        split.SetStretchMax()

        cv = ss.NewClassView("sv")
        cv.AddFrame(split)
        cv.Config()

        tv = gi.AddNewTabView(split, "tv")

        nv = netview.NetView()
        tv.AddTab(nv, "NetView")
        nv.Var = "Act"
        # nv.Params.ColorMap = "Jet" // default is ColdHot
        # which fares pretty well in terms of discussion here:
        # https://matplotlib.org/tutorials/colors/colormaps.html
        nv.SetNet(ss.Net)
        ss.NetView = nv

        nv.Scene().Camera.Pose.Pos.Set(0, 0.87, 2.25)
        nv.Scene().Camera.LookAt(mat32.Vec3(0, -0.16, 0), mat32.Vec3(0, 1, 0))

        plt = eplot.Plot2D()
        tv.AddTab(plt, "TstTrlPlot")
        ss.TstTrlPlot = ss.ConfigTstTrlPlot(plt, ss.TstTrlLog)

        plt = eplot.Plot2D()
        tv.AddTab(plt, "TstEpcPlot")
        ss.TstEpcPlot = ss.ConfigTstEpcPlot(plt, ss.TstEpcLog)

        split.SetSplitsList(go.Slice_float32([.2, .8]))

        recv = win.This()
        
        tbar.AddAction(gi.ActOpts(Label="Init", Icon="update", Tooltip="Initialize everything including network weights, and start over.  Also applies current params.", UpdateFunc=UpdtFuncNotRunning), recv, InitCB)

        tbar.AddAction(gi.ActOpts(Label="Test Trial", Icon="step-fwd", Tooltip="Runs the next testing trial.", UpdateFunc=UpdtFuncNotRunning), recv, TestTrialCB)
        
        tbar.AddAction(gi.ActOpts(Label="Test Item", Icon="step-fwd", Tooltip="Prompts for a specific input pattern name to run, and runs it in testing mode.", UpdateFunc=UpdtFuncNotRunning), recv, TestItemCB)
        
        tbar.AddAction(gi.ActOpts(Label="Test All", Icon="fast-fwd", Tooltip="Tests all of the testing trials.", UpdateFunc=UpdtFuncNotRunning), recv, TestAllCB)

        tbar.AddAction(gi.ActOpts(Label="Stop", Icon="stop", Tooltip="Interrupts running.  Hitting Train again will pick back up where it left off.", UpdateFunc=UpdtFuncRunning), recv, StopCB)
        
        tbar.AddSeparator("log")
        
        tbar.AddAction(gi.ActOpts(Label="README", Icon="file-markdown", Tooltip="Opens your browser on the README file that contains instructions for how to run this model."), recv, ReadmeCB)

        # main menu
        appnm = gi.AppName()
        mmen = win.MainMenu
        mmen.ConfigMenus(go.Slice_string([appnm, "File", "Edit", "Window"]))

        amen = gi.Action(win.MainMenu.ChildByName(appnm, 0))
        amen.Menu.AddAppMenu(win)

        emen = gi.Action(win.MainMenu.ChildByName("Edit", 1))
        emen.Menu.AddCopyCutPaste(win)
        win.MainMenuUpdated()
        vp.UpdateEndNoSig(updt)
        win.GoStartEventLoop()


# TheSim is the overall state for this simulation
TheSim = Sim()

def main(argv):
    TheSim.Config()
    TheSim.ConfigGui()
    TheSim.Init()
    
main(sys.argv[1:])
    
