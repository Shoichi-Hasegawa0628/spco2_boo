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
alpha0 = 12 #20.0 (TC) 
gamma0 = 0.1      #Hyperparameter of CRP in multinomial distribution for index of position distribution
beta0 = 0.1        #Hyperparameter in multinomial distribution P(W) for place names
chi0  = 0.1          #Hyperparameter in multinomial distribution P(φ) for image feature
# k0 = 1e-3          #Hyperparameter in Gaussina distribution P(μ) (Influence degree of prior distribution of μ)
k0 = 0.1      #注意
m0 = np.zeros(dimx)  #Hyperparameter in Gaussina distribution P(μ) (prior mean vector)
V0 = np.eye(dimx)*1  #Hyperparameter in Inverse Wishart distribution P(Σ) (prior covariance matrix)
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
lamb = 0.01

object_dictionary = ['Person', 'Sneakers', 'Chair', 'Other Shoes', 'Hat', 'Car', 'Lamp', 'Glasses', 'Bottle', 'Desk', 'Cup',
        'Street Lights', 'Cabinet/shelf', 'Handbag/Satchel', 'Bracelet', 'Plate', 'Picture/Frame', 'Helmet', 'Book',
        'Gloves', 'Storage box', 'Boat', 'Leather Shoes', 'Flower', 'Bench', 'Potted Plant', 'Bowl/Basin', 'Flag',
        'Pillow', 'Boots', 'Vase', 'Microphone', 'Necklace', 'Ring', 'SUV', 'Wine Glass', 'Belt', 'Monitor/TV',
        'Backpack', 'Umbrella', 'Traffic Light', 'Speaker', 'Watch', 'Tie', 'Trash bin Can', 'Slippers', 'Bicycle',
        'Stool', 'Barrel/bucket', 'Van', 'Couch', 'Sandals', 'Basket', 'Drum', 'Pen/Pencil', 'Bus', 'Wild Bird',
        'High Heels', 'Motorcycle', 'Guitar', 'Carpet', 'Cell Phone', 'Bread', 'Camera', 'Canned', 'Truck',
        'Traffic cone', 'Cymbal', 'Lifesaver', 'Towel', 'Stuffed Toy', 'Candle', 'Sailboat', 'Laptop', 'Awning',
        'Bed', 'Faucet', 'Tent', 'Horse', 'Mirror', 'Power outlet', 'Sink', 'Apple', 'Air Conditioner', 'Knife',
        'Hockey Stick', 'Paddle', 'Pickup Truck', 'Fork', 'Traffic Sign', 'Balloon', 'Tripod', 'Dog', 'Spoon', 'Clock',
        'Pot', 'Cow', 'Cake', 'Dinning Table', 'Sheep', 'Hanger', 'Blackboard/Whiteboard', 'Napkin', 'Other Fish',
        'Orange/Tangerine', 'Toiletry', 'Keyboard', 'Tomato', 'Lantern', 'Machinery Vehicle', 'Fan',
        'Green Vegetables', 'Banana', 'Baseball Glove', 'Airplane', 'Mouse', 'Train', 'Pumpkin', 'Soccer', 'Skiboard',
        'Luggage', 'Nightstand', 'Tea pot', 'Telephone', 'Trolley', 'Head Phone', 'Sports Car', 'Stop Sign',
        'Dessert', 'Scooter', 'Stroller', 'Crane', 'Remote', 'Refrigerator', 'Oven', 'Lemon', 'Duck', 'Baseball Bat',
        'Surveillance Camera', 'Cat', 'Jug', 'Broccoli', 'Piano', 'Pizza', 'Elephant', 'Skateboard', 'Surfboard',
        'Gun', 'Skating and Skiing shoes', 'Gas stove', 'Donut', 'Bow Tie', 'Carrot', 'Toilet', 'Kite', 'Strawberry',
        'Other Balls', 'Shovel', 'Pepper', 'Computer Box', 'Toilet Paper', 'Cleaning Products', 'Chopsticks',
        'Microwave', 'Pigeon', 'Baseball', 'Cutting/chopping Board', 'Coffee Table', 'Side Table', 'Scissors',
        'Marker', 'Pie', 'Ladder', 'Snowboard', 'Cookies', 'Radiator', 'Fire Hydrant', 'Basketball', 'Zebra', 'Grape',
        'Giraffe', 'Potato', 'Sausage', 'Tricycle', 'Violin', 'Egg', 'Fire Extinguisher', 'Candy', 'Fire Truck',
        'Billiards', 'Converter', 'Bathtub', 'Wheelchair', 'Golf Club', 'Briefcase', 'Cucumber', 'Cigar/Cigarette',
        'Paint Brush', 'Pear', 'Heavy Truck', 'Hamburger', 'Extractor', 'Extension Cord', 'Tong', 'Tennis Racket',
        'Folder', 'American Football', 'earphone', 'Mask', 'Kettle', 'Tennis', 'Ship', 'Swing', 'Coffee Machine',
        'Slide', 'Carriage', 'Onion', 'Green beans', 'Projector', 'Frisbee', 'Washing Machine/Drying Machine',
        'Chicken', 'Printer', 'Watermelon', 'Saxophone', 'Tissue', 'Toothbrush', 'Ice cream', 'Hot-air balloon',
        'Cello', 'French Fries', 'Scale', 'Trophy', 'Cabbage', 'Hot dog', 'Blender', 'Peach', 'Rice', 'Wallet/Purse',
        'Volleyball', 'Deer', 'Goose', 'Tape', 'Tablet', 'Cosmetics', 'Trumpet', 'Pineapple', 'Golf Ball',
        'Ambulance', 'Parking meter', 'Mango', 'Key', 'Hurdle', 'Fishing Rod', 'Medal', 'Flute', 'Brush', 'Penguin',
        'Megaphone', 'Corn', 'Lettuce', 'Garlic', 'Swan', 'Helicopter', 'Green Onion', 'Sandwich', 'Nuts',
        'Speed Limit Sign', 'Induction Cooker', 'Broom', 'Trombone', 'Plum', 'Rickshaw', 'Goldfish', 'Kiwi fruit',
        'Router/modem', 'Poker Card', 'Toaster', 'Shrimp', 'Sushi', 'Cheese', 'Notepaper', 'Cherry', 'Pliers', 'CD',
        'Pasta', 'Hammer', 'Cue', 'Avocado', 'Hamimelon', 'Flask', 'Mushroom', 'Screwdriver', 'Soap', 'Recorder',
        'Bear', 'Eggplant', 'Board Eraser', 'Coconut', 'Tape Measure/Ruler', 'Pig', 'Showerhead', 'Globe', 'Chips',
        'Steak', 'Crosswalk Sign', 'Stapler', 'Camel', 'Formula 1', 'Pomegranate', 'Dishwasher', 'Crab',
        'Hoverboard', 'Meat ball', 'Rice Cooker', 'Tuba', 'Calculator', 'Papaya', 'Antelope', 'Parrot', 'Seal',
        'Butterfly', 'Dumbbell', 'Donkey', 'Lion', 'Urinal', 'Dolphin', 'Electric Drill', 'Hair Dryer', 'Egg tart',
        'Jellyfish', 'Treadmill', 'Lighter', 'Grapefruit', 'Game board', 'Mop', 'Radish', 'Baozi', 'Target', 'French',
        'Spring Rolls', 'Monkey', 'Rabbit', 'Pencil Case', 'Yak', 'Red Cabbage', 'Binoculars', 'Asparagus', 'Barbell',
        'Scallop', 'Noddles', 'Comb', 'Dumpling', 'Oyster', 'Table Tennis paddle', 'Cosmetics Brush/Eyeliner Pencil',
        'Chainsaw', 'Eraser', 'Lobster', 'Durian', 'Okra', 'Lipstick', 'Cosmetics Mirror', 'Curling', 'Table Tennis']

# object_dictionary = ['plate', 'bowl', 'pitcher_base', 'banana', 'apple', 'orange', 'cracker_box', 'pudding_box',
#                      'chips_bag', 'coffee', 'muscat', 'fruits_juice', 'pig_doll', 'sheep_doll', 'penguin_doll',
#                      'airplane_toy', 'car_toy', 'truck_toy', 'tooth_paste', 'towel', 'cup', 'treatments', 'sponge',
#                      'bath_slipper', 'dolphin_shaped_sponge', 'flog_shaped_sponge']
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








