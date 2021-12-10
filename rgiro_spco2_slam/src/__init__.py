#coding:utf-8
#This file for setting parameters
#Akira Taniguchi 2017/01/18-2018/02/11-2018/12/22- 
import numpy as np

####################Parameters####################
R = 30               #The number of particles in spatial concept learning (Same to value in run_gmapping.sh)
                     #(It's need to set to the same value in launch file of gmapping: no setting=30)
dimx = 2             #The number of dimensions of xt (x,y)

##Initial (hyper) parameters
##Posterior (∝likelihood×prior): https://en.wikipedia.org/wiki/Conjugate_prior
alpha0 = 20.0        #Hyperparameter of CRP in multinomial distribution for index of spatial concept
gamma0 = 0.1         #Hyperparameter of CRP in multinomial distribution for index of position distribution
beta0 = 0.1          #Hyperparameter in multinomial distribution P(W) for place names 
chi0  = 0.1          #Hyperparameter in multinomial distribution P(φ) for image feature
k0 = 1e-3            #Hyperparameter in Gaussina distribution P(μ) (Influence degree of prior distribution of μ)
m0 = np.zeros(dimx)  #Hyperparameter in Gaussina distribution P(μ) (prior mean vector)
V0 = np.eye(dimx)*2  #Hyperparameter in Inverse Wishart distribution P(Σ) (prior covariance matrix) 
n0 = 3.0             #Hyperparameter in Inverse Wishart distribution P(Σ) {>the number of dimenssions] (Influence degree of prior distribution of Σ)
k0m0m0 = k0*np.dot(np.array([m0]).T,np.array([m0]))


#Parameters of latticelm (Please see web page of latticelm)
knownn = 3#2           #n-gram length for langage model (word n-gram: 3)
unkn   = 3#2           #n-gram length for spelling model (unknown word n-gram: 3)
annealsteps  = 10#3    #焼き鈍し法のステップ数 (3)
anneallength = 15#5    #各焼き鈍しステップのイタレーション数 (5)
burnin   = 100 #10     #burn-inのイタレーション数 (20)
samps    = 100 #10     #サンプルの回数 (100)
samprate = 100 #10     #サンプルの間隔 (1, つまり全てのイタレーション)
ramdoman = 0 #5        #焼きなましパラメータをランダムにする (0:しない、0以外：最大値：各パラメータ値＋randoman) 

#SpCoSLAM 2.0 追加要素
wic = 1             #1:wic重みつき、0:wic重みなし
LAG = 10            #固定ラグ活性化のラグ値(it,ct) (LAG>=1; SpCoSLAM1.0:1)
LMLAG = 10          #LAG #固定ラグ活性化のラグ値(St) (LMLAG>=1), 固定ラグ活性化しない (LMLAG==0) 
tyokuzen = 0        #直前のステップの言語モデルで音声認識 (１) 、ラグ値前の言語モデルで音声認識 (0) 
LMtype = "lattice"  #latticelm:"lattice", lattice→learn→NPYLM:lattice_learn_NPYLM
LMweight = "weight" #wf*ws="weight", P(S{1:t}|c{1:t-1},α,β)/p(S{1:t}|β) = "WS"


####################Option setting (NOT USE)####################
UseFT = 1       #画像特徴を使う場合 (１) 、使わない場合 (０) 
UseLM = 1       #言語モデルを更新する場合 (１) 、しない場合 (０) 

CNNmode = 5     #CNN最終層1000次元(1)、CNN中間層4096次元(2)、PlaceCNN最終層205次元(3)、SIFT(0)

if CNNmode == 0:
  Descriptor = "SIFT_BoF"
  DimImg = 100  #Dimension of image feature
elif CNNmode == 1:
  Descriptor = "CNN_softmax"
  DimImg = 1000 #Dimension of image feature
elif CNNmode == 2:
  Descriptor = "CNN_fc6"
  DimImg = 4096 #Dimension of image feature
elif CNNmode == 3:
  Descriptor = "CNN_Place205"
  DimImg = 205  #Dimension of image feature
elif CNNmode == 4:
  Descriptor = "hybridCNN"
  DimImg = 1183  #Dimension of image feature
elif CNNmode == 5:
  Descriptor = "CNN_Place365"
  DimImg = 365  #Dimension of image feature

#Julius parameters
#Julius folderのsyllable.jconf参照
JuliusVer = "v4.4" #"v.4.3.1" #
HMMtype = "DNN"  #"GMM"
lattice_weight = "AMavg"  #"exp" #Acoustic likelihood (log likelihood: "AMavg", likelihood: "exp")
wight_scale = -1.0
WDs = "0"   #DNN版の単語辞書の音素を*_Sだけにする ("S") , BIE or Sにする ("S"以外) 

if (JuliusVer ==  "v4.4"):
  Juliusfolder = "/home/akira/Dropbox/Julius/dictation-kit-v4.4/"
else:
  Juliusfolder = "/home/akira/Dropbox/Julius/dictation-kit-v4.3.1-linux/"

if (HMMtype == "DNN"):
  lang_init = 'syllableDNN.htkdic' 
else:
  lang_init = 'web.000.htkdic' # 'trueword_syllable.htkdic' #'phonemes.htkdic' # Init dictionary (in ./lang_m/) 
#lang_init_DNN = 'syllableDNN.htkdic' #なごり

#multiCPU for latticelm (a number of CPU)
multiCPU = 1 #2  #max(int(multiprocessing.cpu_count()/2)-1,2) )

##rosbag data playing speed (normal = 1.0)
rosbagSpeed = 1.0 #0.5#2

#latticelm or NPYLM (run SpCoSLAM_NPYLM.sh)
#latticeLM = 1     #latticelm(1), NPYLM(0)
NbestNum = 10 #The number of N of N-best (n<=10) 

####################Setting File PATH####################
#Setting of PATH for output folder
#パスはUbuntu使用時とWin使用時で変更する必要がある。特にUbuntuで動かすときは絶対パスになっているか要確認。
#win:相対パス、ubuntu:絶対パス
datafolder   = "/root/HSR/catkin_ws/src/spco_library/rgiro_spco2_slam/data/output/"        #PATH of data out put folder

speech_folder = "/home/akira/Dropbox/Julius/directory/SpCoSLAM/*.wav" #*.wav" #音声の教示データフォルダ(Ubuntu full path)
speech_folder_go = "/home/akira/Dropbox/Julius/directory/SpCoSLAMgo/*.wav" #*.wav" #評価用の音声データフォルダ(Ubuntu full path)
lmfolder = "/home/akira/Dropbox/SpCoSLAM/learning/lang_m/"

#Folder of training data set (rosbag file)
datasetfolder = "/root/HSR/catkin_ws/src/spco_library/rgiro_spco2_slam/data/rosbag/"   #training data set folder
dataset1      = "albert-b-laser-vision/albert-B-laser-vision-dataset/"
bag1          = "albertBimg.bag"  #Name of rosbag file
datasets      = [dataset1] #[dataset1,dataset2]
bags          = [bag1] #run_rosbag.pyにて使用
scantopic     = ["scan"] #, "base_scan _odom_frame:=odom_combined"]

#dataset2      = "MIT_Stata_Center_Data_Set/"   ##用意できてない
#datasets      = {"albert":dataset1,"MIT":dataset2}
#CNNfolder     = "/home/*/CNN/CNN_Places365/"                        #Folder of CNN model files

#True data files for evaluation (評価用正解データファイル)
correct_Ct = 'Ct_correct.csv'          #データごとの正解のCt番号
correct_It = 'It_correct.csv'          #データごとの正解のIt番号
correct_data = 'SpCoSLAM_human.csv'    #データごとの正解の文章 (単語列、区切り文字つき) (./data/)
correct_data_SEG = 'SpCoSLAM_SEG.csv'  #データごとの正解の文章 (単語列、区切り文字つき) (./data/)
correct_name = 'name_correct.csv'      #データごとの正解の場所の名前 (音素列) 

N_best_number = 10  # The number of N of N-best for PRR evaluation (PRR評価用のN-bestのN)
margin = 10*0.05    # margin value for place area in gird map (0.05m/grid)*margin(grid)=0.05*margin(m)

####################Particle Class (structure)####################
class Particle:
  def __init__(self,id,x,y,theta,weight,pid):
    self.id = id
    self.x = x
    self.y = y
    self.theta = theta
    self.weight = weight
    self.pid = pid
