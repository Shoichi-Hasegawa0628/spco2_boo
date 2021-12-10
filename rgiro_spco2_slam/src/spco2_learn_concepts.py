#!/usr/bin/env python
# -*- coding: utf-8 -*-

##############################################
# SpCoSLAM 2.0 online learning program
# Fixed-lag Rejuvenation of Ct, it, and St
# Re-segmentation of word sequences using NPYLM
# Akira Taniguchi 2017/01/18-2017/02/02-2017/02/15-2017/02/21-2018/02/16-2018/12/22
# Takeshi Nakashima 2021/03/06 
##############################################

# 注意：gmappingと合わせてパーティクル番号の対応関係が合っているか要確認 (評価用の方と合わせて確認が必要)

##########---Finished Implementation 2.0---##########
#[2.0]Additional weight regarding Ct and it for spatial concepts
#[2.0]latticelm→Spatial concept leanring→register result of NPYLM to word dictionary
#[2.0]Calculate and use the weight for language model selection from all St data
#[2.0]Fixed-lag Rejuvenation (Sampling of it and ct) 
#[2.0]Fixed-lag Rejuvenation (Sampling of St, Update language model) 
#Available CPU multiple processing for latticelm
#Available change to GMM or DNN decoder for Julius
#Available change acoustic score to log likelihood or lilelihood for Julius
#Save a file of calculation time
#Calculation of weight using numpy and Bag fix


##########---Process Flow (in gmapping side)---##########
# Particle information of all the time so far (index, self-position coordinates, weight, index at the previous time) is output every frame, file output

# When teaching flag becomes true by spatial concept learning, 
# rosbag is stop
# Process for flag is performed in gmapping
# If weight file is read, go next to codes for resampling.
# Update weights by process of FastSLAM
# Teaching flag is changed to false.
# rosbag is restart


##########---Process Flow (in Python codes)---##########
# The Python program is called from the teaching time flag

### (この処理は前回の単語辞書が得られる時点で先に裏で計算を回しておくのもあり。) 
#JuliusLattice_gmm.pyを呼び出す
##前回の言語モデルを読み込み
##Juliusで教示時刻までの教示データに対して音声認識
##音声認識結果 (ラティス) のファイル形式の変換
##latticelmで単語分割 (パーティクル個数回) 

#画像特徴ftを読み込み (事前に全教示時刻のCNN特徴を得ておく) 

#gmapping側が出力した情報の読み込み
##現在時刻のパーティクル情報 (index,自己位置座標、重み、前時刻のindex) を取得 (ファイル読み込み) 

#過去の教示時刻のパーティクル情報の読み込み
##パーティクルの時刻ごとのindex対応付け処理 (前回の教示のどのパーティクルIDが今回のIDなのか) 
##x_{0:t}を情報整理して得る

#パーティクルごとに計算
##単語情報S_{1:t}、過去のカウント情報C_{1:t-1},i_{1:t-1}、n(l,k),n(l,g),n(l,e),n(k)をファイル読み込み
##it,Ctをサンプリング
### 単語、画像特徴、場所概念index、位置分布indexのカウント数の計算
### スチューデントt分布の計算
##画像の重みwfを計算 (サンプリング時に計算済み) 
##単語の重みwsを計算 (サンプリング時に計算済み) 
##重みの掛け算wt=wz*wf*ws

#重みの正規化wt
#パーティクルIDごとにwtをファイル保存

#パーティクルIDごとに単語情報S_{1:t}、カウント情報C_{1:t},i_{1:t}、n(l,k),n(l,g),n(l,e),n(k)をファイル書き込み

#最大重みのパーティクルの単語情報から単語辞書作成


##########---keep---##########
#[2.0++]事後分布のハイパーパラメータの逐次ベイズ推論
#[2.0++]毎時刻 (m_countが更新される毎) に画像データ取得
#NPYLM(単独使用)との切り替え処理

###Fast SLAMの方の重みが強く効く可能性が高い。→できるだけ毎回場所概念側もリサンプリング？
#ファイル構造の整理


##############################################
import os
import re
import glob
import random
import collections
import numpy as np
import scipy as sp
from numpy.random import multinomial  #,uniform #,dirichlet
#from scipy.stats import t             #,multivariate_normal,invwishart,rv_discrete
#from numpy.linalg import inv, cholesky
from math import pi as PI
from math import cos,sin,sqrt,exp,log,fabs,fsum,degrees,radians,atan2,gamma,lgamma
#from sklearn.cluster import KMeans
from multiprocessing import Pool
from multiprocessing import Process
import multiprocessing
from __init__ import *
from spco2_math import *
import csv # Takeshi Nakashima 2021/03/06 
import rospy
from std_msgs.msg import String
import std_msgs.msg

import sys
import os.path
import time
from __init__ import * # SyntaxWarning を無視


# Reading data for image feature (NOT USE)
def ReadImageData(trialname, step):
  FT = []
  for s in xrange(step):
    for line in open( datafolder + trialname + '/img/ft' + str(s+1) + '.csv', 'r'):
    #for line in open( datasetfolder + datasetname + 'img/ft' + str(s+1) + '.csv', 'r'):
      itemList = line[:].split(',')
    FT.append( [float(itemList[i]) for i in xrange(DimImg)] )
  return FT

# Reading data for image feature
#時刻情報を取得して、一番近い画像特徴ファイルを読み込むようにする
def ReadImageData2(trialname, datasetname, step, clocktime):
  '''
   return Image Feature list at all teaching step
  '''
  clocktime = int(clocktime)
  
  print ("step=",step)
  #time.sleep(5)
  FT = []
  for s in xrange(step-1):
    savefilepath = datafolder + trialname + '/img/ft' + str(s+1) + '.csv'
    print (savefilepath)
    for line in open( datafolder + trialname + '/img/ft' + str(s+1) + '.csv', 'r'):
      itemList = line[:].split(',')
      print ("itemList=",itemList)
      print ("s=",s)
      #time.sleep(5)
    FT.append( [float(itemList[i]) for i in xrange(DimImg)] )
  
  # Read image data at current time
  
  files = glob.glob(datasetfolder + datasetname + Descriptor + "/*.csv")
  files.sort()
  #print "files:",files
  files2 = [files[i][len(datasetfolder)+len(datasetname)+len(Descriptor)+1:-4] for i in xrange(len(files))]
  #print "files2:",files2
  #print "files2:"
  #time.sleep(5)
  #print files2
  #temptime = []
  #for f in files:
  #  m = re.match('[^\d]*(\d+).*$', f)
  #  temptime.append(int(m.groups[0]))
  temptime = [int(re.match('[^\d]*(\d+).*$', f).groups()[0]) for f in files2]
  #print "temptime:",temptime
  #print "temptime:"
  #print "clocktime:",clocktime
  #time.sleep(5)
  if (clocktime in temptime):
    f_temp = files[temptime.index(clocktime)]
  else:
    if (clocktime+1 in temptime):
      f_temp = files[temptime.index(clocktime)]
    elif(clocktime-1 in temptime):
      f_temp = files[temptime.index(clocktime)]
    else:
      print ("FT read error.", clocktime, temptime)
      f_temp = files[-1]
  print ("f_temp:",f_temp)
  #time.sleep(5)
  for line in open( f_temp, 'r'):
      itemList = line[:].split(',')
  FT.append( [float(itemList[i]) for i in xrange(DimImg)] )
  
  # Write new data file
  fp = open( datafolder + trialname + '/img/ft' + str(step) + '.csv', 'w')
  for i in xrange(DimImg):
    fp.write(repr(float(itemList[i]))+',')
  fp.close()
  
  # Save time information of image file 
  fp = open( datafolder + trialname + '/img/time' + str(step) + '.csv', 'w')
  fp.write(repr(f_temp))
  fp.close()
  
  return FT

# Reading word data and Making word list
def ReadWordData(step, trialname, particle):
  N = 0
  Otb = []
  #WordSegList = []
  Otb_FilePath= '/root/HSR/catkin_ws/src/spco_library/rgiro_spco2_slam/data/output/test/tmp/Otb.csv'
  ######################################################
  #固定ラグ活性化の場合の処理
  if (LMLAG != 0):
    tau = step - LMLAG
    max_particle =1 #2021/03/06 
    if (tau >= 1):
      #最大重みのパーティクル番号を読み込む
      if (LMweight != "WS"):
        omomi = '/weights.csv'
      else: #if (LMweight == "WS"):
        omomi = '/WS.csv'
      
      i = 0
      for line in open(datafolder + trialname + '/'+ str(tau) + omomi, 'r'):   ##読み込む
          #itemList = line[:-1].split(',')
          if (i == 0):
            max_particle = int(line)
          i += 1
  ######################################################
  
  #テキストファイルを読み込み
  with open(Otb_FilePath, 'r') as f:
        reader = csv.reader(f)
        Otb = [row for row in reader]

  ##場所の名前の多項分布のインデックス用
  W_list = []
  for n in xrange(len(Otb)):
    for j in xrange(len(Otb[n])):
      if ( (Otb[n][j] in W_list) == False ):
        W_list.append(Otb[n][j])
        #print str(W_list),len(W_list)
  
  ##時刻tデータごとにBOW化(?)する、ベクトルとする
  Otb_BOW = [ [0 for i in xrange(len(W_list))] for n in xrange(len(Otb)) ]
  
  for n in xrange(len(Otb)):
    for j in xrange(len(Otb[n])):
      for i in xrange(len(W_list)):
        if ( W_list[i] == Otb[n][j] ):
          Otb_BOW[n][i] = Otb_BOW[n][i] + 1
  
  ###################################################### #2021/03/06 
  #固定ラグ活性化の場合の処理(samp.100を更新)
  #if (LMLAG != 0) and (tau >= 1):
  #  fp = open( filename + '/out_gmm/' + str(particle) + '_samp.'+str(samps), 'w')
  #  for n in xrange(N):
  #    fp.write(str(WordSegList[n])+"\n")
  #  #fp.write("\n")
  #  fp.close()
    
  ######################################################
  
  #Output W_list and Otb_BoW 2021/03/06 
  if (particle == max_particle):
    print ("Otb:")
    print(Otb)
    print ("Otb_BOW:")
    print(Otb_BOW)
    print ("W_list:")
    print(W_list)

    ## save massage as a csv format
    FilePath= datafolder + trialname + "/tmp/step"+ str(step)  +"_Otb.csv"
    with open(FilePath, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(Otb)

    fOtb = open( datafolder + trialname + "/tmp/step"+ str(step)  +"_Otb.txt" , "w" )
    for i in range(len(Otb)):
      for j in range(len(Otb[i])):
        fOtb.write(str(Otb[i][j])+" ")
      fOtb.write("\n")
    fOtb.close()

    #fOtb_BOW = open( datafolder + trialname + "/tmp/step"+ str(step) +"_particle"+ str(particle) +"_Otb_BOW.txt" , "w" )
    FilePath= datafolder + trialname + "/tmp/step"+ str(step)  +"_Otb_BOW.csv"
    with open(FilePath, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(Otb_BOW)

    FilePath= datafolder + trialname + "/tmp/step"+ str(step)  +"_W_list.csv"
    with open(FilePath, 'w') as f:
        writer = csv.writer(f,lineterminator='\n')
        writer.writerow(W_list)

  return W_list, Otb_BOW, Otb


#itとCtのデータを読み込む (教示した時刻のみ) 
def ReaditCtData(trialname, step, particle):
  CT,IT = [],[]
  if (step != 1):  #最初のステップ以外
    for line in open( datafolder + trialname + "/" + str(step-1) + "/particle" + str(particle) + ".csv" , 'r' ):
      itemList = line[:-1].split(',')
      CT.append( int(itemList[7]) )
      IT.append( int(itemList[8]) )
  return CT, IT

# Reading particle data (ID,x,y,theta,weight,previousID)
def ReadParticleData(m_count, trialname):
  p = []
  for line in open ( datafolder + trialname + "/particle/" + str(m_count) + ".csv" ):
    itemList = line[:-1].split(',')
    p.append( Particle( int(itemList[0]), float(itemList[1]), float(itemList[2]), float(itemList[3]), float(itemList[4]), int(itemList[5])) )
  return p


# パーティクルIDの対応付け処理(Ct,itの対応付けも)
def ParticleSearcher(trialname):
  m_count = 0  #m_countの数
  #Count m_count that is number of gmapping output steps
  while (os.path.exists( datafolder + trialname + "/particle/" + str(m_count+1) + ".csv" ) == True):
    m_count += 1
  
  #if (m_count == 0):  #エラー処理
  #  #m_countのindexは1から始まる
  #  while (os.path.exists( datafolder + trialname + "/particle/" + str(m_count+1) + ".csv" ) == True):
  #   m_count += 1
  
  if (m_count == 0):  #エラー処理:Set teachingflag zero and exit spco2_learn_concepts.py process.
    print ("m_count",m_count)
    flag = 0
    fp = open( datafolder + trialname + "/teachingflag.txt", 'w')
    fp.write(str(flag))
    fp.close()
    exit()
  
  #教示された時刻のみのデータにする
  steplist = m_count2step(trialname, m_count)
  step = len(steplist)
  #print steplist
  
  #C[1:t-1],I[1:t-1]のパーティクルID(step-1時点)と現在のparticleIDの対応付け
  CTtemp = [[] for r in xrange(R)]
  ITtemp = [[] for r in xrange(R)]
  #Read spatial_concept_index(Ct) and position_distributions_index(It) results of the privious teaching step
  for particle in xrange(R):
    CTtemp[particle],ITtemp[particle] = ReaditCtData(trialname, step, particle)
  #print "CTtemp,ITtemp:",CTtemp,ITtemp
  
  p = [[Particle( int(0), float(1), float(2), float(3), float(4), int(5) ) for i in xrange(R)] for c in xrange(m_count)]
  #Read trajectory infomation (ID,x,y,theta,weight,previousID) of all gmapping step
  for c in xrange(m_count):
    p[c] = ReadParticleData(c+1, trialname)        #m_countのindexは1から始まる   
    ######非効率なので、前回のパーティクル情報を使う (未実装) 
  #print "p:",p
  p_trajectory = [ [Particle( int(0), float(1), float(2), float(3), float(4), int(5) ) for c in xrange(m_count)] for i in xrange(R) ]
  CT = [ [0 for s in xrange(step-1)] for i in xrange(R) ]
  IT = [ [0 for s in xrange(step-1)] for i in xrange(R) ]
  
  for i in xrange(R):
    c_count = m_count -1  #一番最後の配列から処理
    #print c_count,i
    p_trajectory[i][c_count] = p[c_count][i]
    
    # If you get error message for CT, IT in first step, please remove these 4 lines comment-out (#372-#375).
    if (step == 1): ##CT,IT is empty list in old version.
      CT[i] = CTtemp[i]
      IT[i] = ITtemp[i]
    elif (step == 2): ##step==1のときの推定値を強制的に1にする
      CT[i] = [1]
      IT[i] = [1]
    elif (steplist[-2][0] == m_count): #m_countが直前のステップにおいても同じ場合の例外処理
      print ("m_count", steplist[-2][0], steplist[-1][0])
      CT[i] = [ CTtemp[i][s] for s in xrange(step-1)]
      IT[i] = [ ITtemp[i][s] for s in xrange(step-1)]
    
    for c in xrange(m_count-1):  #0～最後から2番目の配列まで
      preID = p[c_count][p_trajectory[i][c_count].id].pid
      p_trajectory[i][c_count-1] = p[c_count-1][preID]
      
      if (step != 1) and (step != 2) and (steplist[-2][0] != m_count):
       if (steplist[-2][0] == c_count): #CTtemp,ITtempを現在のパーティクルID順にする
          CT[i] = [ CTtemp[preID][s] for s in xrange(step-1)]
          IT[i] = [ ITtemp[preID][s] for s in xrange(step-1)]
          #print i,preID
          #print i, c, c_count-1, preID
      c_count -= 1
  
  X_To = [ [Particle( int(0), float(1), float(2), float(3), float(4), int(5) ) for c in xrange(step)] for i in xrange(R) ]
  for i in xrange(R):
    #preID = p[m_count-1][p_trajectory[i][c_count].id].pid
    #test
    X_To[i] = [p_trajectory[i][steplist[s][0]-1] for s in xrange(step)] ##steplist[s][0]-1はsでよいのでは？->ダメ
  
  return X_To, step, m_count, CT, IT


#gmappingの時刻カウント数 (m_count) と教示時刻のステップ数 (step) を対応付ける
def m_count2step(trialname, m_count):
  list= []  #[ [m_count, step], ... ]
  step = 1
  csvname = datafolder + trialname + "/m_count2step.csv"
  
  if ( os.path.exists( csvname ) != True ):  ##ファイルがないとき、作成する
    fp = open( csvname, 'w')
    fp.write("")
    fp.close()
  else:
    for line in open ( csvname , 'r'):
      itemList = line[:-1].split(',')
      #print itemList
      list.append( [int(itemList[0]), int(itemList[1])] )
      step += 1
  
  
  #Update m_count2step.csv
  if (step == len(list)+1) or (step == 1 and len(list) == 1):  #テスト用の実行で同じm_countのデータカウントが増えないように
    fp = open( csvname, 'a')
    fp.write(str(m_count) + "," + str(step))
    fp.write('\n')
    fp.close()
    list.append( [m_count, step] )
  
  return list


#パーティクル情報の保存
def WriteParticleData(filename, step, particle, Xp, p_weight, ct, it, CT, IT):
  cstep = step - 1
  #ID,x,y,theta,weight,pID,Ct,it
  fp = open( filename + "/particle" + str(particle) + ".csv", 'w')
  for s in xrange(step-1):
      fp.write( str(s) + "," + str(Xp[s].id) + "," + str(Xp[s].x) + "," + str(Xp[s].y) + "," + str(Xp[s].theta) + "," + str(Xp[s].weight) + "," + str(Xp[s].pid) + "," + str(CT[s]) + "," + str(IT[s]) )
      fp.write('\n')
  fp.write( str(cstep) + "," + str(Xp[cstep].id) + "," + str(Xp[cstep].x) + "," + str(Xp[cstep].y) + "," + str(Xp[cstep].theta) + "," + str(p_weight) + "," + str(Xp[cstep].pid) + "," + str(ct) + "," + str(it) )
  fp.write('\n')

#パーティクルごとに単語情報を保存
def WriteWordData(filename, particle, W_list_i):
  fp = open( filename + "/W_list" + str(particle) + ".csv", 'w')
  for w in xrange(len(W_list_i)):
    fp.write(W_list_i[w]+",")
  fp.close()

#重み (log) を保存 (gmapping読み込み用) 
def WriteWeightData(trialname, m_count, p_weight_log):
  fp = open( datafolder + trialname + "/weight/" + str(m_count) + ".csv", 'w')
  for r in xrange(R):
    fp.write(str(p_weight_log[r])+",")
  fp.close()

#パーティクルごとのCtとitのインデックス対応を保存
def WriteIndexData(filename, particle, ccitems, icitems,ct,it):
  fp = open( filename + "/index" + str(particle) + ".csv", 'w')
  for c in xrange(len(ccitems)):
    fp.write(str(ccitems[c][0])+",")
  if ( ct == (max(ccitems)[0]+1) ):
    fp.write(str(ct) + ",")
  fp.write("\n")
  for i in xrange(len(icitems)):
    fp.write(str(icitems[i][0])+",")
  #if ( it == (max(icitems)[0]+1) ):
  #  fp.write(str(it) + ",")
  fp.write("\n")
  fp.close()

# Saving data for parameters Θ of spatial concepts
def SaveParameters(filename, particle, phi, pi, W, theta, mu, sig):
  fp = open( filename + "/phi" + str(particle) + ".csv", 'w')
  for i in xrange(len(phi)):
    for j in xrange(len(phi[i])):
      fp.write(repr(phi[i][j])+",")
    fp.write('\n')
  fp.close()
  
  fp2 = open( filename + "/pi" + str(particle) + ".csv", 'w')
  for i in xrange(len(pi)):
    fp2.write(repr(pi[i])+",")
  fp2.write('\n')
  fp2.close()
  
  fp3 = open( filename + "/W" + str(particle) + ".csv", 'w')
  for i in xrange(len(W)):
    for j in xrange(len(W[i])):
      fp3.write(repr(W[i][j])+",")
    fp3.write('\n')
  fp3.close()
  
  fp4 = open( filename + "/theta" + str(particle) + ".csv", 'w')
  for i in xrange(len(theta)):
    for j in xrange(len(theta[i])):
      fp4.write(repr(theta[i][j])+",")
    fp4.write('\n')
  fp4.close()
  
  fp5 = open(filename + "/mu" + str(particle) + ".csv", 'w')
  for k in xrange(len(mu)):
      for dim in xrange(dimx):
        fp5.write(repr(mu[k][dim])+',')
      fp5.write('\n')
  fp5.close()
  
  fp6 = open(filename + "/sig" + str(particle) + ".csv", 'w')
  for k in xrange(len(sig)):
      for dim in xrange(dimx):
        for dim2 in xrange(dimx):
          fp6.write(repr(sig[k][dim][dim2])+',')
        #fp6.write('\n')
      fp6.write('\n')
  fp6.close()

# Online Learning for Spatial Concepts of one particle
def Learning(step, filename, particle, XT, ST, W_list, CT, IT, FT):
    #Update ct, it, weight_log, WS_log
    #  St:Bag of word representation of every teaching sentence.
    #  W_list[i]:word dictionary for particle(i).
    #  ST_seq[i]:plain text of every teaching sentence
    XT_list = [ np.array([XT[s].x, XT[s].y]) for s in range(step) ]
    np.random.seed()
    ########################################################################
    ####                   　    ↓Learning phase↓                       ####
    ########################################################################
    print (u"- <START> Learning of Spatial Concepts in Particle:" + str(particle) + " -")
    ##sampling ct and it
    print (u"Sampling Ct,it...")
    cstep = step - 1
    #k0m0m0 = k0*np.dot(np.array([m0]).T,np.array([m0]))
    G = len(W_list)
    E = DimImg
    
    
    if (step == 1):  #初期設定
      ct = 1
      it = 1
      ccitems = [(1,1)]
      icitems = [(1,1)]
      
      phi = [np.array([1])]
      pi = np.array([1])
      
      #Cの場所概念にStのカウントを足す
      Nlg_c = [ sum( [np.array(ST[0])] ) ]
      #Wc_temp = [(np.array(Nlg_c[0]) + beta0 ) / (sum(Nlg_c[0]) + G*beta0)]
      W = [(np.array(Nlg_c[0]) + beta0 ) / (sum(Nlg_c[0]) + G*beta0)] #[Wc_temp[0]]
      
      #Cの場所概念にFtのカウントを足す
      Nle_c = [ sum( [np.array(FT[0])] ) ]
      #thetac_temp = [(np.array(Nle_c[0]) + chi0 ) / (sum(Nle_c[0]) + E*chi0)]
      theta = [(np.array(Nle_c[0]) + chi0 ) / (sum(Nle_c[0]) + E*chi0)] #[thetac_temp[0]]
      print ("theta=",theta)
      print ("theta(arg)",(np.array(Nle_c[0]) + chi0 ) / (sum(Nle_c[0]) + E*chi0))
      #time.sleep(30)

      #nk = 1
      kN,mN,nN,VN = PosteriorParameterGIW2(1,1,1,[0],XT_list[0],0)
      """
      xk = [np.array([XT[0].x, XT[0].y])]
      m_ML = sum(xk) #/ float(nk) #fsumではダメ
      kN = k0 + nk
      mN = ( k0*m0 + nk*m_ML ) / kN  #dim 次元横ベクトル
      nN = n0 + nk
      VN = V0 + sum([ np.dot( np.array([xk[0]]).T,np.array([xk[0]]) ) ]) + k0m0m0 - kN*np.dot(np.array([mN]).T,np.array([mN])) 
      """
      MU = [mN]
      SIG = [np.array(VN) / (nN - dimx - 1)]
      weight_log = log(1.0/R) #0.0 #XT[cstep].weight
      WS_log = log(1.0/R) ###################
      
    else:
      #CTとITのインデックスは0から順番に整理されているものとする->しない
      cc = collections.Counter(CT) #｛Ct番号：カウント数｝
      L = len(cc)   #場所概念の数
      
      ic = collections.Counter(IT) #｛it番号：カウント数｝
      K = len(ic)   #位置分布の数
      
      ccitems = cc.items()  # [(ct番号,カウント数),(),...]
      cclist = [ccitems[c][1] for c in xrange(L)] #各場所概念ｃごとのカウント数
      
      icitems = ic.items()  # [(it番号,カウント数),(),...]
      iclist = [icitems[i][0] for i in xrange(K)] #各位置分布iごとのindex番号
      
      #場所概念のindexごとにITを分解
      ITc = [[] for c in xrange(L)]
      #c = 0
      for s in xrange(cstep):
        for c in xrange(L):
          if (ccitems[c][0] == CT[s]):  ##場所概念のインデックス集合ccに含まれるカテゴリ番号のみ
            #print s,c,CT[s],ITc[c]
            ITc[c].append(IT[s])
            #c += 1
      icc = [collections.Counter(ITc[c]) for c in xrange(L)] #｛cごとのit番号：カウント数｝の配列
      #Kc = [len(icc[c]) for c in xrange(L)]  #場所概念ｃごとの位置分布の種類数≠カウント数
      
      icclist = [ [icc[c][iclist[i]] for i in xrange(K)] for c in xrange(L)] #場所概念ｃにおける各位置分布iごとのカウント数
      #print np.array(icclist)
      
      Nct   = sum(cclist) #場所概念の総カウント数=データ数(前ステップ:t-1)
      #Nit_l = sum(np.array(iccList))  #場所概念ｃごとの位置分布の総カウント数＝ 場所概念ｃごとのカウント数
      #これはcclistと一致するはず
      print (Nct,cclist) #, Nit_l
      
      CRP_CT  = np.array(cclist + [alpha0]) / (Nct + alpha0)
      CRP_ITC = np.array([ np.array([ (icclist[c][i]*(icclist[c][i] != 0) + gamma0*(icclist[c][i] == 0)) for i in xrange(K)] + [gamma0]) / (cclist[c] + gamma0) for c in xrange(L) ] + [ np.array([0.0 for i in xrange(K)] + [1.0]) ])
      #for c in xrange(L):
      #  for i in xrange(K):
      #    if (icclist2[c][i] == 0):
      #      icclist2[c][i] = gamma0
      #CRP_ITC = np.array( [np.array([icclist2[c][i] for i in xrange(K)] + [gamma0]) / (cclist[c] + gamma0) for c in xrange(L)] + [np.array([0.0 for i in xrange(K)] + [1.0])] )
      
      #print Xp
      xt = XT_list[cstep] #np.array([XT[cstep].x, XT[cstep].y])
      tpdf = np.array([1.0 for i in xrange(K+1)])
      
      for k in xrange(K+1):
        #データがあるかないか関係のない計算
        #事後t分布用のパラメータ計算
        if (k == K):  #k is newの場合
          nk = 0
          kN,mN,nN,VN = PosteriorParameterGIW2(k,nk,step-1,IT,XT_list,0)
        else:
          nk = ic[icitems[k][0]]  #icitems[k][1]
          kN,mN,nN,VN = PosteriorParameterGIW2(k,nk,step-1,IT,XT_list,icitems[k][0])
        
        #t分布の事後パラメータ計算
        mk = mN
        dofk = nN - dimx + 1
        InvSigk = (VN * (kN +1)) / (kN * dofk)
        
        #ｔ分布の計算
        #logt = log_multivariate_t_distribution( xt, mu, Sigma, dof)  ## t(x|mu, Sigma, dof)
        tpdf[k] = multivariate_t_distribution( xt, mk, InvSigk, dofk )  ## t(x|mu, Sigma, dof)
        #print "tpdf",k,tpdf
        
      
      #ctとitの組をずらっと横に並べる (ベクトル) ->2次元配列で表現 (temp[L+1][K+1]) [L=new][K=exist]は0
      #temp2 = np.array([[10.0**10 * tpdf[i] * CRP_ITC[c][i] * CRP_CT[c] for i in xrange(K+1)] for c in xrange(L+1)])
      print ("--------------------") #####
      #print temp2 #####
      temp22 = 10.0**10 * tpdf.T * (CRP_CT*CRP_ITC.T).T #####
      #temp2 = temp22
      print (temp22) #####
      print ("--------------------") #####
      
      St_prob = np.array([1.0 for c in xrange(L+1)])
      Ft_prob = np.array([1.0 for c in xrange(L+1)])
      Bt = sum(ST[cstep]) #発話文中の単語数
      
      #位置分布kの計算と場所概念cの計算を分ける(重複計算を防ぐ)
      for l in xrange(L+1):
          if (l < L):
            ##単語のカウント数の計算
            #STはすでにBOWなのでデータstepごとのカウント数になっている
            Nlg = sum([np.array(ST[s])*(CT[s]==ccitems[l][0]) for s in xrange(step-1)])  #sumだとできる
            W_temp_log = np.log(np.array(Nlg) + beta0 ) - np.log(sum(Nlg) + G*beta0)
            St_prob[l] = np.exp(sum(np.array(W_temp_log) * np.array(ST[cstep])))
            
            ##画像特徴のカウント数の計算
            Nle = sum([np.array(FT[s])*(CT[s]==ccitems[l][0]) for s in xrange(step-1)])  #sumだとできる
            theta_temp_log = np.log(np.array(Nle) + chi0 ) - np.log(sum(Nle) + E*chi0)
            Ft_prob[l] = np.exp(sum(np.array(theta_temp_log) * np.array(FT[cstep]))) #.prod() #要素積
          else:  #ct=lかつit=kのデータがない場合
            St_prob[l] = 1.0/(G**Bt)
            Ft_prob[l] = 1.0/E  ##画像特徴は全次元足して１になるのでこれで良い
          
          #temp2[l] = temp2[l] * St_prob[l] * Ft_prob[l]
      temp2 = (temp22.T * St_prob * Ft_prob).T
      print (temp2)
      
      #2次元配列を1次元配列にする
      c,i = 0,0
      temp = []
      cxi_index_list = []
      for v in temp2:
        i = 0
        for item in v:
          temp.append(item)
          cxi_index_list.append((c,i))   #一次元配列に2次元配列のindexを対応付け
          i += 1
        c += 1
      
      temp = np.array(temp) / np.sum(temp)  #正規化
      #print temp
      cxi_index = list(multinomial(1,temp)).index(1)
      
      #1次元配列のindexを2次元配列に戻す (Ctとitに分割する) 
      C,I = cxi_index_list[cxi_index]
      #print C,I
      
      
      ###################### 場所概念パラメータΘの更新の事前準備
      Kp = K
      Lp = L
      #ct,itがNEWクラスターなら新しい番号を与える
      if C == L:
        ct = max(cc)+1
        print ("c="+ str(ct) +" is new.")
        Lp += 1
      else:
        ct = ccitems[C][0]
      if I == K:
        it = max(ic)+1
        print ("i="+ str(it) +" is new.")
        Kp += 1
        icitems += [(it,1)]
      else:
        it = icitems[I][0]
      
      #print ct,it
      print ("C", C, ", ct", ct, "; I", I, ", it", it)
      
      #C,Kのカウントを増やす
      if C == L:  #L is new
        cclist.append(1)
        ccitems.append((ct,1))  ##ccitemsも更新する
        if I == K:  #K is new
          for c in xrange(L):
            icclist[c].append(0)  #既存の各場所概念ごとに新たな位置分布indexを増やす
        icclist.append([1*(k==I) for k in xrange(Kp)])  #新たな場所概念かつ新たな位置分布のカウント
      else:  #L is exist
        cclist[C] += 1
        if I == K:  #K is new
          for c in xrange(L):
            icclist[c].append(0)  #既存の各場所概念ごとに新たな位置分布indexを増やす
        icclist[C][I] += 1
      
      ic[icitems[I][0]] += 1

      CT.append( ct )
      IT.append( it )
      
      ##############################################################
      #Fixed-lag Rejuvenation
      if (LAG != 0):
        ###カウント数が追加されたデータをもとに、ラグ値ごとにリサンプリング
        for lag in xrange(LAG-1):
          tau = cstep - LAG + 1 + lag
          if (tau >= 0):
            ###tau = t - LAG + 1 +lagのカウントを除く
            CT[tau] = -1
            IT[tau] = -1
            
            #CTとITのインデックスは0から順番に整理されているものとする->しない
            cc = collections.Counter(CT) #｛Ct番号：カウント数｝
            cc.pop(-1)
            L = len(cc)   #場所概念の数
            
            ic = collections.Counter(IT) #｛it番号：カウント数｝
            ic.pop(-1)
            K = len(ic)   #位置分布の数
            
            ccitems = cc.items()  # [(ct番号,カウント数),(),...]
            cclist = [ccitems[c][1] for c in xrange(L)] #各場所概念ｃごとのカウント数
            
            icitems = ic.items()  # [(it番号,カウント数),(),...]
            iclist = [icitems[i][0] for i in xrange(K)] #各位置分布iごとのindex番号
            
            #場所概念のindexごとにITを分解
            ITc = [[] for c in xrange(L)]
            #c = 0
            for s in xrange(step):
              for c in xrange(L):
                if (ccitems[c][0] == CT[s]):  ##場所概念のインデックス集合ccに含まれるカテゴリ番号のみ
                  #print s,c,CT[s],ITc[c]
                  ITc[c].append(IT[s])
                  #c += 1
            icc = [collections.Counter(ITc[c]) for c in xrange(L)] #｛cごとのit番号：カウント数｝の配列
            icclist = [ [icc[c][iclist[i]] for i in xrange(K)] for c in xrange(L)] #場所概念ｃにおける各位置分布iごとのカウント数
            #print np.array(icclist)
            
            Nct   = sum(cclist) #場所概念の総カウント数=データ数(現ステップから一つ除いたもの)
            print (Nct,cclist) #, Nit_l
            
            ###サンプリング式を再計算
            CRP_CT  = np.array(cclist + [alpha0]) / (Nct + alpha0)
            CRP_ITC = np.array([ np.array([ (icclist[c][i]*(icclist[c][i] != 0) + gamma0*(icclist[c][i] == 0)) for i in xrange(K)] + [gamma0]) / (cclist[c] + gamma0) for c in xrange(L) ] + [ np.array([0.0 for i in xrange(K)] + [1.0]) ])
            
            xt = XT_list[tau] #np.array([XT[tau].x, XT[tau].y])
            tpdf = np.array([1.0 for i in xrange(K+1)])
            
            for k in xrange(K+1):
              #データがあるかないか関係のない計算
              #事後t分布用のパラメータ計算
              if (k == K):  #k is newの場合
                nk = 0
                kN,mN,nN,VN = PosteriorParameterGIW2(k,nk,step,IT,XT_list,0)
              else:
                nk = ic[icitems[k][0]]  #icitems[k][1]
                kN,mN,nN,VN = PosteriorParameterGIW2(k,nk,step,IT,XT_list,icitems[k][0])
              
              #t分布の事後パラメータ計算
              mk = mN
              dofk = nN - dimx + 1
              InvSigk = (VN * (kN +1)) / (kN * dofk)
              
              #ｔ分布の計算
              #logt = log_multivariate_t_distribution( xt, mu, Sigma, dof)  ## t(x|mu, Sigma, dof)
              tpdf[k] = multivariate_t_distribution( xt, mk, InvSigk, dofk )  ## t(x|mu, Sigma, dof)
              #print "tpdf",k,tpdf
            
            
            #2次元配列で表現 (temp[L+1][K+1]) [L=new][K=exist]は0
            #temp2 = np.array([[10.0**10 * tpdf[i] * CRP_ITC[c][i] * CRP_CT[c] for i in xrange(K+1)] for c in xrange(L+1)])
            print ("--------------------") #####
            #print temp2 #####
            temp22 = 10.0**10 * tpdf.T * (CRP_CT*CRP_ITC.T).T #####
            #temp2 = temp22
            print (temp22) #####
            print ("--------------------") #####
            
            St_prob = np.array([1.0 for c in xrange(L+1)])
            Ft_prob = np.array([1.0 for c in xrange(L+1)])
            Bt = sum(ST[tau]) #発話文中の単語数
            
            #位置分布kの計算と場所概念cの計算を分ける(重複計算を防ぐ)
            for l in xrange(L+1):
                if (l < L):
                  ##単語のカウント数の計算
                  #STはすでにBOWなのでデータstepごとのカウント数になっている
                  Nlg = sum([np.array(ST[s])*(CT[s]==ccitems[l][0]) for s in xrange(step)])  #sumだとできる
                  W_temp_log = np.log(np.array(Nlg) + beta0 ) - np.log(sum(Nlg) + G*beta0)
                  St_prob[l] = np.exp(sum(np.array(W_temp_log) * np.array(ST[tau])))
                  
                  ##画像特徴のカウント数の計算
                  Nle = sum([np.array(FT[s])*(CT[s]==ccitems[l][0]) for s in xrange(step)])  #sumだとできる
                  theta_temp_log = np.log(np.array(Nle) + chi0 ) - np.log(sum(Nle) + E*chi0)
                  Ft_prob[l] = np.exp(sum(np.array(theta_temp_log) * np.array(FT[tau]))) #.prod() #要素積
                else:  #ct=lかつit=kのデータがない場合
                  St_prob[l] = 1.0/(G**Bt)
                  Ft_prob[l] = 1.0/E  ##画像特徴は全次元足して１になるのでこれで良い
                
                #temp2[l] = temp2[l] * St_prob[l] * Ft_prob[l]
            temp2 = (temp22.T * St_prob * Ft_prob).T
            print (temp2)
            
            ###リサンプリング
            #2次元配列を1次元配列にする
            c,i = 0,0
            temp = []
            cxi_index_list = []
            for v in temp2:
              i = 0
              for item in v:
                temp.append(item)
                cxi_index_list.append((c,i))   #一次元配列に2次元配列のindexを対応付け
                i += 1
              c += 1
            
            temp = np.array(temp) / np.sum(temp)  #正規化
            #print temp
            cxi_index = list(multinomial(1,temp)).index(1)
            
            #1次元配列のindexを2次元配列に戻す (Ctとitに分割する) 
            C_tau,I_tau = cxi_index_list[cxi_index] 
            #サンプリングされた番号と実際のindex番号は異なることに注意
            #print C,I
            
            ###サンプリングされた値を追加
            Kp = K
            Lp = L
            #ct,itがNEWクラスターなら新しい番号を与える
            if C_tau == L:
              ct = max(cc)+1
              print ("c="+ str(ct) +" is new.")
              Lp += 1
            else:
              ct = ccitems[C_tau][0]
            if I_tau == K:
              it = max(ic)+1
              print ("i="+ str(it) +" is new.")
              Kp += 1
              icitems += [(it,1)]
            else:
              it = icitems[I_tau][0]
            
            print ("C_tau", C_tau, ", ct", ct, "; I_tau", I_tau, ", it", it)
            
            
            #C_tau,I_tauのカウントを増やす
            if C_tau == L:  #L is new
              cclist.append(1)
              ccitems.append((ct,1))  ##ccitemsも更新する
              if I_tau == K:  #K is new
                for c in xrange(L):
                  icclist[c].append(0)  #既存の各場所概念ごとに新たな位置分布indexを増やす
              icclist.append([1*(k==I_tau) for k in xrange(Kp)])  #新たな場所概念かつ新たな位置分布のカウント
            else:  #L is exist
              cclist[C_tau] += 1
              if I_tau == K:  #K is new
                for c in xrange(L):
                  icclist[c].append(0)  #既存の各場所概念ごとに新たな位置分布indexを増やす
              icclist[C_tau][I_tau] += 1
            
            #for k in xrange(Kp):
            #  if(k == I_tau):  
            ic[icitems[I_tau][0]] += 1
            #    #print icitems[k][0], ic[icitems[k][0]]
            #  #print k,nk

            CT[tau] =  ct
            IT[tau] =  it
      
      ##############################################################
      
      #場所概念パラメータΘの更新処理
      #pi = (np.array(cclist) + alpha0) / (Nct+1 + alpha0*Lp)  #np.array(iccList) /  (Nct + alpha0)
      #phi = [(np.array(icclist[c]) + gamma0) / (cclist[c] + gamma0*Kp) for c in xrange(Lp)]
      pi = (np.array(cclist) + alpha0/float(Lp)) / (Nct+1 + alpha0)  #np.array(iccList) /  (Nct + alpha0)
      phi = [(np.array(icclist[c]) + gamma0/float(Kp)) / (cclist[c] + gamma0) for c in xrange(Lp)]
      
      #Cの場所概念にStのカウントを足す
      Nlg_c = [sum([np.array(ST[s])*(CT[s]==ccitems[c][0]) for s in xrange(step)]) for c in xrange(Lp)] 
      #Wc_temp = [(np.array(Nlg_c[c]) + beta0 ) / (sum(Nlg_c[c]) + G*beta0) for c in xrange(Lp)]
      W = [(np.array(Nlg_c[c]) + beta0 ) / (sum(Nlg_c[c]) + G*beta0) for c in xrange(Lp)] #[Wc_temp[c] for c in xrange(Lp)]
      
      #Cの場所概念にFtのカウントを足す
      Nle_c = [sum([np.array(FT[s])*(CT[s]==ccitems[c][0]) for s in xrange(step)]) for c in xrange(Lp)] 
      #thetac_temp = [(np.array(Nle_c[c]) + chi0 ) / (sum(Nle_c[c]) + E*chi0) for c in xrange(Lp)]
      theta = [(np.array(Nle_c[c]) + chi0 ) / (sum(Nle_c[c]) + E*chi0) for c in xrange(Lp)] # [thetac_temp[c] for c in xrange(Lp)]
      
      #Iの位置分布にxtのカウントを足す
      mNp = [[] for k in xrange(Kp)]
      nNp = [[] for k in xrange(Kp)]
      VNp = [[] for k in xrange(Kp)]
      
      for k in xrange(Kp):
        nk = ic[icitems[k][0]]
        #if(k == I_tau):  ##既存クラスのとき
        #  nk = nk + 1
        #  #print icitems[k][0], ic[icitems[k][0]]
        #print k,nk
        
        kN,mN,nN,VN = PosteriorParameterGIW2(k,nk,step,IT,XT_list,icitems[k][0])

        mNp[k] = mN
        nNp[k] = nN
        VNp[k] = VN
      """
      if (I_tau == K): #新規クラスがあるとき(ない場合もある)
        #nk = 1
        #print I,nk
        kN,mN,nN,VN = PosteriorParameterGIW(I_tau,1,1,[0],[XT[cstep]],0)

        mNp[K] = mN
        nNp[K] = nN
        VNp[K] = VN
      """
      MU = mNp #[mNp[k] for k in xrange(Kp)]
      SIG = [np.array(VNp[k]) / (nNp[k] - dimx - 1) for k in xrange(Kp)]
      
      
      ############### 重みの計算 ###############
      wz_log = XT[cstep].weight
      
      ##############################################################
      #Wic = P(X{t}|X{1:t-1},c{1:t-1},i{1:t-1})の計算
      if (wic == 1):
        sum_itct = np.sum( temp22 )#sum( [CRP_ITC[c][i] * CRP_CT[c] for c in xrange(L+1)] )
        wic_log = np.log(sum_itct)
      ##############################################################
      
      #P(Ft|F{1:t},c{1:t-1},α,χ)の計算
      wf = np.sum(Ft_prob * CRP_CT) #sum( [Ft_prob[c] * CRP_CT[c] for c in xrange(L+1)] )
      wf_log = np.log(wf)
      #print wf, wf_log
      
      #P(St|S{1:t},c{1:t-1},α,β)の計算
      psc = np.sum(St_prob * CRP_CT)#sum( [St_prob[c] * CRP_CT[c] for c in xrange(L+1)] )
      if (UseLM == 1):
        #単なる単語の生起確率 (頻度+βによるスムージング) :P(St|S{1:t-1},β)
        Ng = sum([np.array(ST[s]) for s in xrange(step-1)])  #sumだとできる
        W_temp2_log = np.log(np.array(Ng) + beta0 ) - log(sum(Ng) + G*beta0)
        ps_log = sum(np.array(W_temp2_log) * np.array(ST[cstep]))
        
        ws_log = np.log(psc) - ps_log
      else:
        ws_log = np.log(psc)
      #print log(psc), ps_log, ws_log
      
      ##############################################################
      #P(S{1:t}|c{1:t-1},α,β)/p(S{1:t}|β)の計算
      WS_log = ws_log
      if (LMweight == "WS"):
        #ct_1hot = [ [1*(c==l) for c in xrange(L+1)] for l in xrange(L+1)]
        #P_SlC_log = log(psc) #ws_log
        for i in xrange(step-1):
          step_iter = step-1-i  #stepを逆にたどって計算する
          #print step_iter, step-1,len(ST),len(CT)
          if (step_iter != 1):  #p(S1|β)はパーティクルによらず同じ確率のため計算を省く
            Nlg_step = sum([np.array(ST[s])*(CT[s]==CT[step_iter-1]) for s in xrange(step_iter-1)])
            #print step_iter, Nlg_step
            W_temp_log_step = np.log(np.array(Nlg_step) + beta0 ) - log(sum(Nlg_step) + G*beta0)
            St_logprob_step = sum(np.array(W_temp_log_step) * np.array(ST[step_iter]))
            WS_log  += St_logprob_step
            
            #単なる単語の生起確率 (頻度+βによるスムージング) :P(S{1:t-1}|β)
            Ng_step = sum([np.array(ST[s]) for s in xrange(step_iter-1)])  #sumだとできる
            W_temp2_log_step = np.log(np.array(Ng_step) + beta0 ) - log(sum(Ng_step) + G*beta0)
            ps_log_step = sum(np.array(W_temp2_log_step) * np.array(ST[step_iter]))
            WS_log -= ps_log_step
        print ("WS_log:",WS_log)
      ##############################################################
      
      #weight_log = wz_log+wf_log+ws_log   #sum of log probability
      weight_log = wf_log+ws_log   #sum of log probability
      if (wic == 1):
        weight_log += wic_log
        print ("wic_log:", wic_log)
      print (wz_log,wf_log,ws_log, weight_log, np.exp(weight_log))
      #print weight_log, np.exp(weight_log)
      
    ########################################################################
    ####      　                 ↑Learning phase↑                       ####
    ########################################################################
    loop = 1
    ########  ↓File Output of Learning Result↓  ########
    if loop == 1:
        #最終学習結果を出力(ファイルに保存)
        SaveParameters(filename, particle, phi, pi, W, theta, MU, SIG)
        ###実際のindex番号と処理上のindex番号の対応付けを保存 (場所概念パラメータΘは処理上の順番) 
        WriteIndexData(filename, particle, ccitems, icitems,ct,it)  
    ########  ↑File Output of Learning Result↑  ########
    #print "--------------------"
    print (u"- <COMPLETED> Learning of Spatial Concepts in Particle:" + str(particle) + " -")
    #print "--------------------"
    return ct, it, weight_log, WS_log


##############################並列化##############################
"""
def multiCPU_latticelm(taple):
      #from __init__ import * # SyntaxWarning を無視
      i = int(taple[1])
      filename = taple[0]
      #パーティクルごとに計算
      #for i in xrange(R):
      print ("--------------------------------------------------")
      print ("Particle:",i)
      
      if (ramdoman != 0):
        annealsteps  += random.randint(-1*annealsteps+1,ramdoman)
        anneallength += random.randint(-1*anneallength+1,ramdoman)
      
      #print "latticelm run. sample_num:" + str(sample)
      latticelm_CMD = "latticelm -input fst -filelist "+ filename + "/fst_gmm/fstlist.txt -prefix " + filename + "/out_gmm/" + str(i) + "_ -symbolfile " + filename + "/fst_gmm/isyms.txt -burnin " + str(burnin) + " -samps " + str(samps) + " -samprate " + str(samprate) + " -knownn " + str(knownn) + " -unkn " + str(unkn) + " -annealsteps " + str(annealsteps) + " -anneallength " + str(anneallength)
      
      p = os.popen( latticelm_CMD )   ##latticelm ###################
      #time.sleep(1.0) #sleep(秒指定)###################
      p.close()###########################
      latticelm_count = 0
      while (os.path.exists(filename + "/out_gmm/" + str(i) + "_samp."+str(samps) ) != True):
              print (filename + "/out_gmm/" + str(i) + "_samp."+str(samps),"wait(10s)... or ERROR?")
              #p.close()
              latticelm_count += 1
              if (latticelm_count > 10):
                print ("run latticelm again.")
                p = os.popen( latticelm_CMD )   ##latticelm 
                p.close()
                latticelm_count = 0
                time.sleep(1.0) #sleep(秒指定)
      #p.close()###########################
      print ("Particle:",i," latticelm complete!")

##############################並列化##############################
"""
########################################
#if __name__ == '__main__':
def callback(message): 
    
    trialname = rospy.get_param('~trial_name') 
    datasetNUM = rospy.get_param('~dataset_NUM') 
    print("Start_Learning")
    #time.sleep(5)

    print (trialname)
    datasetname = datasets[int(datasetNUM)]
    print (datasetname) #datasetPATH
    #print trialname
    start_iter_time = time.time()
    #Read following 
    #Xp:particle history(ID,x,y,theta,weight,previousID) at all gmapping step."
    #step:counter of teaching
    #m_count:counter of gmapping step
    #Ct:spatial_concept_index of privious teaching
    #It:position_distributions_index of privious teaching
    Xp,step,m_count,CT,IT = ParticleSearcher(trialname)
    for i in xrange(R): #この例外処理はなんのためにある？
      while (0 in CT[i]) or (0 in IT[i]):
        print ("Error! 0 in CT,IT",CT,IT)
        Xp,step,m_count,CT,IT = ParticleSearcher(trialname)
    
    print ("step", step)
    print ("m_count", m_count)
    print ("Xp:particle_trajectory_and_weight_transition", Xp)
    print ("Ct:spatial_concept_index", CT)
    print ("It:position_distributions_index",IT)
    #time.sleep(30)

    #teachingtime = []
    #for line in open( datasetfolder + datasetname + 'teaching.csv', 'r'):
      #itemList = line[:].split(',')
    #  teachingtime.append(float(line))
    
    #clocktime = float(teachingtime[step-1]) ##
    
    # Request output file name
    #filename = raw_input("trialname?(folder) >")
    filename = datafolder + trialname + "/" + str(step)  ##FullPath of learning trial folder
    Makedir( filename ) 
    
    print ("filename",filename)
    print ("trialname",trialname)

    p_weight_log = np.array([0.0 for i in xrange(R)])
    p_weight = np.array([0.0 for i in xrange(R)])
    p_WS_log = np.array([0.0 for i in xrange(R)]) ###
    W_list   = [[] for i in xrange(R)]
    ST_seq   = [[] for i in xrange(R)]
    
    if (UseFT == 1):
      #FT = ReadImageData2(trialname, datasetname, step, clocktime)
      FT = ReadImageData(trialname, step)
    else:
      FT = [[0 for e in xrange(DimImg)] for s in xrange(step)]
    """    2021/03/06 
    p = Pool(multiCPU) #max(int(multiprocessing.cpu_count()/2)-1,2) )
    taple = [(filename,i) for i in xrange(R)]
    p.map(multiCPU_latticelm, taple)  #パーティクルごとに並列計算
    """    
    #パーティクルごとに計算
    for i in xrange(R):
      
      #for i in xrange(R): ###############
      print ("--------------------------------------------------") ###############
      print ("Particle:",i) ###############
      
      #Read teaching infomation
      #  St:Bag of word representation of every teaching sentence.
      #  W_list[i]:word dictionary for particle(i).
      #  ST_seq[i]:plain text of every teaching sentence
      W_list[i], ST, ST_seq[i] = ReadWordData(step, trialname, i)

      print ("Read word data.")
      #CT, IT = ReaditCtData(trialname, step, i)
      print ("Read Ct,It data.")
      print ("CT",CT[i])
      print ("IT",IT[i])
      ct, it, p_weight_log[i],p_WS_log[i] = Learning(step, filename, i, Xp[i], ST, W_list[i], CT[i], IT[i], FT)     ## Learning of spatial concepts
      print ("Particle:",i," Learning complete!")
      
      WriteParticleData(filename, step, i, Xp[i], p_weight_log[i], ct, it, CT[i], IT[i])  #重みは正規化されてない値が入る
      WriteWordData(filename, i, W_list[i])
      
      print ("Write particle data and word data.")
    
    print ("--------------------------------------------------") ###############
    #logの最大値を引く処理
    #print p_weight_log
    logmax = np.max(p_weight_log)
    p_weight_log = p_weight_log - logmax  #np.arrayのため
    
    WriteWeightData(trialname, m_count, p_weight_log)
    
    #print p_weight_log
    #weightの正規化
    p_weight = np.exp(p_weight_log)
    sum_weight = np.sum(p_weight)
    p_weight = p_weight / sum_weight
    print ("Weight:",p_weight)
    
    #########################################################################
    #logの最大値を引く処理
    #print p_WS_log
    logmax = np.max(p_WS_log)
    p_WS_log = p_WS_log - logmax  #np.arrayのため
    
    #weightの正規化
    p_WS = np.exp(p_WS_log)
    sum_WS = np.sum(p_WS)
    p_WS = p_WS / sum_WS
    print ("WS:",p_WS)
    #########################################################################
    
    MAX_weight_particle = np.argmax(p_weight) #p_weight.index(max(p_weight))                     ##最大重みのパーティクルindex
    MAX_WS_particle = np.argmax(p_WS) #p_weight.index(max(p_weight))                     ##最大重みのパーティクルindex
    if (LMweight == "WS"):
      MAX_LM_particle = MAX_WS_particle
      
      ##最大重みのパーティクルindexと重みを保存する
      fp = open(filename + "/WS.csv", 'w')
      fp.write(str(MAX_WS_particle) + '\n')
      for particle in xrange(R):
        fp.write(str(p_WS[particle]) + ',')
      fp.write('\n')
      fp.close()
    else:
      MAX_LM_particle = MAX_weight_particle
    
    print (MAX_LM_particle)
    
    ##########最大重みのパーティクルの単語集合についてNPYLMを実行##########
    if (LMtype == "lattice_learn_NPYLM"):
      wordDic = [] #set()
      W_list_particle = []
      ST_seq = ST_seq[MAX_LM_particle]
      ##事前処理 (音節の間に空白を挿入) 
      if (tyokuzen == 1) and (LMLAG != 0) and (step-LMLAG >= 1): # and (tyokuzenNPYLM == 1) これがあってもエラーなく動いていた？？（tyokuzenNPYLMが未定義）
        ST_seq_step_LMLAG = ST_seq[MAX_LM_particle][0:step-LMLAG]
        ST_seq = ST_seq[step-LMLAG:step] #[ [ST_seq[n][j] for j in xrange(len(ST_seq[n]))] for n in xrange(step-LMLAG, len(ST_seq))]

        #W_listを更新
        for n in xrange(step-LMLAG):   ##*_samp.*を読み込む
          for j in xrange(len(ST_seq_step_LMLAG[n])):
            if ( (ST_seq_step_LMLAG[n][j] in W_list_particle) == False ):
              W_list_particle.append(ST_seq_step_LMLAG[n][j])

      #print ST_seq
      for n in xrange(len(ST_seq)):
        ST_seq_temp = "" #[]
        word2 = ""
        for j in xrange(len(ST_seq[n])):
          ST_seq_temp += ST_seq[n][j].decode('sjis')
        for w in xrange(len(ST_seq_temp)): #空白をいれる処理
          #if ((word[w] or word[w].encode('sjis')) == ("ぁ" or "ぃ" or "ぅ" or "ぇ" or "ぉ" or "ゃ" or "ゅ" or "ょ" or u'ぁ' or u'ぃ' or u'ぅ' or u'ぇ' or u'ぉ' or u'ゃ' or u'ゅ' or u'ょ')):   # or or word[w].encode('sjis') or word[w]
          word2 = word2 + ST_seq_temp[w] + u" "
          #print u'ぁ',"ぁ",word[w]
          #else:[:-1]
          #  word2 = word2 + word[w] + u" "
        word2 = word2.replace(u"  ", u" ")
        word2 = word2.replace(u" ぁ", u"ぁ")
        word2 = word2.replace(u" ぃ", u"ぃ")
        word2 = word2.replace(u" ぅ", u"ぅ")
        word2 = word2.replace(u" ぇ", u"ぇ")
        word2 = word2.replace(u" ぉ", u"ぉ")
        word2 = word2.replace(u" ゃ", u"ゃ")
        word2 = word2.replace(u" ゅ", u"ゅ")
        word2 = word2.replace(u" ょ", u"ょ")
        
        if (word2 != ""):
          #print type(word2) #= #unicode(word2, 'shift_jis')
          #print len(word2)
          wordDic.append( word2 )
          #print wordDic
          print (len(wordDic),word2)
      
      #NPYLMのためのテキストファイルを保存
      f = open( filename + "/out_gmm/sentences" + str(MAX_LM_particle) + ".char" , "w")# , "sjis" )
      for w2 in range(len(wordDic)):
        f.write(wordDic[w2].encode('sjis')) #
        f.write('\n')
      f.close()
      
      #NPYLMを実行
      latticelm_CMD2 = "latticelm -input text -prefix " + filename + "/out_gmm/NPYLM" + str(MAX_LM_particle) + "_ -burnin " + str(burnin) + " -samps " + str(samps) + " -samprate " + str(samprate) + " -knownn " + str(knownn) + " -unkn " + str(unkn) + " -annealsteps " + str(annealsteps) + " -anneallength " + str(anneallength) + " " + filename + "/out_gmm/sentences" + str(MAX_LM_particle) + ".char"
      p = os.popen( latticelm_CMD2 )   ##latticelm ###################
      #time.sleep(1.0) #sleep(秒指定)###################
      p.close()###########################

      print ("NPYLM complete!")
      #Otb = []
      #W_list_particle = []
      #W_listを更新
      for line in open( filename + '/out_gmm/NPYLM' + str(MAX_LM_particle) + '_samp.'+str(samps), 'r'):   ##*_samp.*を読み込む
        itemList = line[:-1].split(' ')
        while ("" in itemList):
          itemList.pop(itemList.index(""))
        #Otb = Otb + [itemList]
        
        for j in xrange(len(itemList)):
          if ( (itemList[j] in W_list_particle) == False ):
            W_list_particle.append(itemList[j])
    else:
      W_list_particle = W_list[MAX_LM_particle]
    ##########最大重みのパーティクルの単語集合についてNPYLMを実行##########
    
    #W_list_particle = W_list[MAX_LM_particle]
    #WordDictionaryUpdate(step, filename, W_list_particle)       ##単語辞書登録 
    #print "Language Model update!"
    
    ##最大重みのパーティクルindexと重みを保存する
    fp = open(filename + "/weights.csv", 'w')
    fp.write(str(MAX_weight_particle) + '\n')
    for particle in xrange(R):
      fp.write(str(p_weight[particle]) + ',')
    fp.write('\n')
    fp.close()
    
    #処理終了のフラグを送る
    #endflag = "1"
    #pub.publish(endflag)
    
    flag = 0
    fp = open( datafolder + trialname + "/teachingflag.txt", 'w')
    fp.write(str(flag))
    fp.close()
    
    fp = open( datafolder + trialname + "/gwaitflag.txt", 'w')
    fp.write(str(m_count+1))
    fp.close()
    
    end_iter_time = time.time()
    iteration_time = end_iter_time - start_iter_time
    fp = open( datafolder + trialname + "/time_step.txt", 'a')
    fp.write(str(step)+","+str(iteration_time)+"\n")
    fp.close()
########################################

    ## Publish messeage for start_visualization
    str_msg = 'start_visualization'#std_msgs.msg.String(data= message.data )
    print ("Publish!")
    pub.publish(str_msg)

if __name__ == '__main__':
  rospy.init_node('SpCoSLAM', anonymous=False)
  print("ready to learning")
  sub = rospy.Subscriber('start_learning', String, callback)
  pub = rospy.Publisher('start_visualization', std_msgs.msg.String, queue_size=10,latch=True)
  rospy.spin()
