#coding:utf-8
#This file for setting parameters
#Akira Taniguchi 2017/01/18-2018/02/11-2018/12/22- 
import numpy as np
import roslib.packages
import os

####################Parameters####################
R = 30               #The number of particles in spatial concept learning (Same to value in run_gmapping.sh)
                     #(It's need to set to the same value in launch file of gmapping: no setting=30)
dimx = 2             #The number of dimensions of xt (x,y)

##Initial (hyper) parameters
##Posterior (∝likelihood×prior): https://en.wikipedia.org/wiki/Conjugate_prior
# alpha0 = 20.0        #Hyperparameter of CRP in multinomial distribution for index of spatial concept
alpha0 = 18.9 #18.9(Ex1) # 0.09(site Visit)
gamma0 = 0.1         #Hyperparameter of CRP in multinomial distribution for index of position distribution
beta0 = 0.1          #Hyperparameter in multinomial distribution P(W) for place names 
chi0  = 0.1          #Hyperparameter in multinomial distribution P(φ) for image feature
# k0 = 1e-3          #Hyperparameter in Gaussina distribution P(μ) (Influence degree of prior distribution of μ)
k0 = 1.0 # 1(ex1)           #注意
m0 = np.zeros(dimx)  #Hyperparameter in Gaussina distribution P(μ) (prior mean vector)
V0 = np.eye(dimx)*2  #Hyperparameter in Inverse Wishart distribution P(Σ) (prior covariance matrix) 
n0 = 3.0             #Hyperparameter in Inverse Wishart distribution P(Σ) {>the number of dimenssions] (Influence degree of prior distribution of Σ)
k0m0m0 = k0*np.dot(np.array([m0]).T,np.array([m0]))

#SpCoSLAM 2.0 追加要素
wic = 1             #1:wic重みつき、0:wic重みなし
LAG = 10            #固定ラグ活性化のラグ値(it,ct) (LAG>=1; SpCoSLAM1.0:1)
LMLAG = 10          #LAG #固定ラグ活性化のラグ値(St) (LMLAG>=1), 固定ラグ活性化しない (LMLAG==0) 
tyokuzen = 0        #直前のステップの言語モデルで音声認識 (１) 、ラグ値前の言語モデルで音声認識 (0) 

#LMtype = "lattice"  #latticelm:"lattice", lattice→learn→NPYLM:lattice_learn_NPYLM
LMweight = "weight" #wf*ws="weight", P(S{1:t}|c{1:t-1},α,β)/p(S{1:t}|β) = "WS"

#SpCoSLAM (Bag-Of-Objects追加バージョン)
lamb = 0.1
object_dictionary = ["plate", "bowl", "pitcher_base", "banana",
                      "apple", "orange", "cracker_box", "pudding_box",
                      "chips_bag", "coffee", "muscat", "fruits_juice",
                      "pig_doll", "sheep_doll", "penguin_doll", "airplane_toy",
                      "car_toy", "truck_toy", "tooth_paste", "towel",
                      "cup", "treatments", "sponge", "bath_slipper"]
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

####################Setting File PATH####################
#Setting of PATH for output folder
#パスはUbuntu使用時とWin使用時で変更する必要がある。特にUbuntuで動かすときは絶対パスになっているか要確認。
#win:相対パス、ubuntu:絶対パス
datafolder   = "/root/HSR/catkin_ws/src/spco2_boo/rgiro_spco2_slam/data/output/"        #PATH of data out put folder

####################Particle Class (structure)####################
class Particle:
  def __init__(self,id,x,y,theta,weight,pid):
    self.id = id
    self.x = x
    self.y = y
    self.theta = theta
    self.weight = weight
    self.pid = pid

# SPCO_PARAM_PATH = str(roslib.packages.get_pkg_dir("rgiro_spco2_slam")) + "/data/output/test/max_likelihood_param/"








