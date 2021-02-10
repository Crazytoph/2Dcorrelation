# 2Dcorrelation

Last Edit: February, 2021

Project for my master thesis. The aim is to perform 2D correlation analysis and various data pretreatment. Open for any help, feedback or suggestions.
Some parts are build on certain file-naming conventions and the data output of the measurement devices of our group.

contact: crazytoph@hotmail.de


Modules:
-------
main, hotznplots, analise and cdata

cdata:
  cdata contains the class CData, which defines the object attributes of the data of one measurement. The goal is to simple create objects for each measurement through    the folder path and then access simple key attributes directly from the object if wanted use for other purpose e.g. figures.
  
hotznplots: 
  hotznplots contains all plotting functions. Right now there exist 2 Types, Heatmap-Plot and normal 1D Function-Plot. We try to unify plots to minimize necessary input parameters while still give the possibility to adapt plots to different purposes.
 
analise:
  Under andalise we find all analytic tools. One big tool planned is 2D correlation analysis. 
  
main:
  The main-module needs to be run in order to start the program. However, currently we plan to work in an Interactive Python Shell (IDE) and import only the modules needed. In this case, the main module can be neglected.
  
  
Small Step-by-Step Introductions:
--------------------------------

Plotting with IDE of Spyder:
  Spyder has an advanced interactive editor and is integrated in Anaconda. To creat plot with spyder do folowing:
  
  1. Make sure Spyder knows the program folder. Go  Tools -> PYTHONPATH Manager and if not yet done ADD the path of the module .py files.
  2. start by importing the modules in the Editor:          import cdata, import hotznplots as plot
  3. get the path of the files, e.g:                        path = input() -> Copy in Path
  4. create object:                                         data = cdata.CData(path)
  5. plot, e.g.:                                            plot.function(data.t_df)
  
  For parameters see documentation of the single functions() in the docstring. Note for the *args or *df parameter, manye DataFrames can be given by
  "plot.function(data1.t_df, data2.t_df...)" to plot same wavelengths of different data. For heatmap maximum amount is 2 and plotted into 2 subplots.
  For funtion "plot.function(data.t_df, df2=data2.t_df) creates 2 subplots.
  
Working with Git:
  1. Open Git Bash
  2. Go to Git folder -> cd F:/GitHub/2Dcorrelation
  3. check status in main branch: git checkout master
  4. if not up-to-date:     git pull
  5. merge to coop-branch   checkout coop
                            git merge master
  5. if implement changes:  git checkout coop 
                            work 
                            git checkout
                            if files changed, they appear red
                            git add "file"
                            git commit -m "change description"
                            git push
                            log-in 2 times 
                            
  Note Git commands should also work fine in Spyder, if Git is installed like !git command...
  
                            
  
