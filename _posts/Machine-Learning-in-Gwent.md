---
title: 'Machine Learning in Gwent / 昆特牌的机器学习打法'
date: 2022-05-25
permalink: /posts/2012/08/blog-post-4/
tags:
  - Gwent
  - Machine Learning
  - 品万卷杂文
---

###前言

前些日子颇不务正业，玩了玩一款休闲小游戏，名叫《小杰来帮忙》，又名《巫师三：狂猎》(The Witcher III : Wild Hunt)。
![巫师三：游戏截图](https://upload-images.jianshu.io/upload_images/13641295-31ed4a4db3e067a2.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


这个游戏可能并不是非常有名，任务节奏非常的悠闲，小杰 (Geralt) 在聊天喝茶的时候，总是喜欢跟他的朋友们玩玩昆特牌 (Gwent) 。这个卡牌类游戏倒是引起了我的兴趣，即便是在大战前的紧要关头，也会抓紧时间，先和旅店老板打两局牌再走。可是，后期的几场大赛总是不能如意，尤其是在“大赌局” (High Stakes) 中与莎莎 (Sasha) 夫人的一场比赛，险象环生，挣扎良久才勉强过关，让我义愤难平。在不停的摸索中，我的打法逐渐发生了变化，从起先的快攻到后来的间谍流，再往后则采纳了欲扬先抑的策略。几经转变打法，我不禁思考，作为一个博弈游戏，昆特牌是否有其最优策略？如果有其最优策略，我们是否能够用合理的手段把它寻找出来呢？作为一个非常不熟悉机器学习 (Machine Learning) 的“机器学习助教”，我自然而然的反应便是：能不能设计一款机器学习的软件，让机器自行发掘出昆特牌的最优打法。问题是，这个真的可以实现吗？答案当然是肯定的，毕竟很多年前，人类博弈游戏最高峰的围棋已经被 AlphaGo 所征服，对于这样一个人人都可以上手的卡牌类游戏，想必其机器学习算法也并不复杂。

![昆特牌示意图](https://upload-images.jianshu.io/upload_images/13641295-a877da45d2bd7a1b.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


于是乎趁几日闲来无事，我便着手去做一个昆特牌的机器学习打法教程。但是鉴于鄙人才疏学浅，编程水平尤为欠缺，想要一个人去实现全部的昆特牌内容，恐怕有些困难，故自作主张，将昆特牌规则进行了一定程度的简化。现将简版规则介绍如下：

###昆特牌简化规则

1. 昆特牌为回合制对战类卡牌游戏，每场比赛采取三局两胜制。
2. 比赛开场前双方各持10张手牌，每张牌左上角数字标注其战斗力。
3. 每一局分为多个回合，每一回合双方各需打出一张手牌，否则视为本局比赛弃权，即本局比赛不能再继续出牌。
4. 每一局比赛，当双方均放弃出牌时进行结算，场上全部牌的战斗力总和大的一方获得该局游戏的胜利。
5. 每一局比赛结束后，场上已打出的牌将全部进入弃牌堆，在接下来的比赛中不可继续使用。

值得注意的是，三局比赛的中途无法获得新的手牌，当某名玩家第一局打出过多手牌后，他后面两局将无法抵挡对方的强大攻势，导致其最终失败。另外，由于我们想要找到的是一个最优的出牌策略，而不是一味地凭借稀有的高战斗力卡牌碾压对手获得胜利，我决定设计一个较为公平的初始手牌情况：即双方各持10张手牌，手牌的战斗力从1点至10点，每种各一张。因此，如何巧妙地运用自己有限的手牌，采取田忌赛马的策略，获得胜利，是一个并不简单的博弈论问题。

以上就是对昆特牌规则的简要介绍，基于如此的规则，我们开始设计昆特牌机器学习的框架。鉴于本人学术不精，代码中如有纰漏，还望各位大神多多指正，不胜感激。

注：相较于原版的昆特牌，在这里我们简化掉了各种特殊技能牌（如间谍，医生，号角等）、天气影响以及领导牌，主要原因是这些牌的技术实现稍为复杂，有兴趣和能力的读者，可以自行补充这部分代码。

###面向对象的游戏设计框架
一场比赛 (Game) 由三局 (Round) 构成，一局又有若干回合 (Turn)，而游戏的操纵者是两名玩家 (Player)，游戏的道具是诸多卡牌 (Card)。这一连串的角色共同组成了一场完整的游戏。因此，如果要编辑一场游戏，我们的首要任务是理清各个角色之间的关系，并且将他们的关系通过程序表达出来。以此为出发点，我们会发现面向对象的编程方式正符合我们的需求，而Python作为计算机语言中面向对象编程的典范，自然成为我们的首选语言。Python发展到现在，有很多的函数已经封装成包，我们将在程序中调用以下包来方便函数实现：

```
import random  # 随机数
import copy    # 对象拷贝

# 常用数据处理包
from pandas import Series, DataFrame
import pandas as pd
import numpy as np

# csv文件处理包
import csv

# 机器学习包
from sklearn import linear_model
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
```

面向对象的编程，即在程序中定义一个个不同的类，每个类拥有其自己的独特属性和操作类型，在类之下定义独立的对象，我们可以通过调用对象的函数或修改对象的属性实现各种操作。

例如在游戏中，显然两名玩家在操作方面有很强的共性，我们即可为他们定义一个类(Class)，即“玩家”类，并在该类下面定义两个独立的对象(Object)：玩家一与玩家二。“玩家”都拥有共同的属性，如每名玩家都有手牌数、生命值等，而不同玩家的这些属性对应的值却可以各不相同。例如，比赛中的某一时刻，玩家一和玩家二都有手牌，而玩家一的手牌数为3张，玩家二的手牌数为4张，那么“手牌”是“玩家”类的属性，而“3张”和“4张”分别是玩家一和玩家二对应属性的值。

另外，我们可以在“玩家”类下定义相应的操作，例如“玩家”可以“摸牌”，而每名玩家摸牌后只影响自己的手牌数，并不影响其他玩家的手牌数。在这里，我们可以将“摸牌”操作定义为“玩家”的“类函数”，而每名玩家可以通过调用自身对象的对应函数，来改变自己的手牌数。

以上，我们对面向对象的程序设计有了最基础的认识，就可以着手分析一局游戏的具体步骤，并开始写我们的程序了。


###对象设计
##### 1. 卡牌
卡牌 (Card) 是游戏的最基础元素，也是我们的第一个需要定义的类。每张卡牌拥有自己的序号，名称，战斗力以及其他附属属性。在这里，我们只定义最基础的战士卡牌，并生成战斗力为1-10的10个对象。

```
class Card:
    empCount = 0 # 总牌数
 
    # 构造函数，用来定义对象
    def __init__(self, index, name, power):
        self.index = index    # 序号
        self.name = name    # 名称
        self.power = power  # 战斗力
        Card.empCount += 1
   
    # 展示卡牌总计
    def displayCount(self):
        print("Total Card %d" % Card.empCount)
 
    # 展示单张卡牌
    def displayCard(self):
        print("Index: ", self.index, "\t Name:", self.name,  "\t Power: ", self.power)
```

```
# 定义10个卡牌对象
card1 = Card(1, "Pawn\t", 1)
card2 = Card(2, "Guard\t", 2)
card3 = Card(3, "Infantry\t", 3)
card4 = Card(4, "Knight\t", 4)
card5 = Card(5, "Elite\t", 5)
card6 = Card(6, "Rook\t", 6)
card7 = Card(7, "Bishop\t", 7)
card8 = Card(8, "Catapult\t", 8)
card9 = Card(9, "Triss\t", 9)
card10 = Card(10, "Geralt\t", 10)

# 将卡牌加入卡牌池
cardPool = [card1, card2, card3, card4, card5, card6, card7, card8, card9, card10]
```



##### 2. 场域
游戏中，卡牌可以位于不同的场域 (Board)，例如牌组、手牌、场上以及弃牌堆。这些场域具备一些相同的特性，例如都可以容纳、增加或剔除卡牌。
```
class Board:
    
    # 构造函数，用来定义对象
    def __init__(self, name):
        self.cards = []
        self.power = 0
        self.count = 0
        self.name = name
    
    # 增加
    def add(self, newCard):
        self.cards.append(newCard)
        self.power += newCard.power
        self.count += 1
        
    # 剔除
    def out(self, outCard):
        for card in self.cards.copy():
            if card.index == outCard.index:
                #print("True")
                self.cards.remove(card)
                self.power -= card.power
                self.count -= 1
                return True
        return False
                
    # 展示
    def display(self):
        print("=== ", self.name, " ===")
        print("Count:", self.count)
        print("Power:", self.power)
        for card in self.cards:
            card.displayCard()
        print("=== End ", self.name, " ===")
```



##### 3. 玩家
玩家是游戏的主角，每个玩家都会以己方获胜为目标，通过一系列操作，左右游戏的进程。因此，对于玩家操作的定义在我们编程过程中至关重要。

######玩家的基础属性
- 姓名或编号


- 场域

为简便起见，我们给每一个玩家分配以下场域：牌组 (Deck)、手牌 (Hand)、部署(Deploy) 和弃牌堆 (Graveyard)。

- 弃权标记

弃权标记 (Pass Signal)，是一个布尔型0-1变量，若为真 (True)，则表明玩家当前局弃权。

- 生命值

生命值 (Health)，是一个整数型变量，初始玩家生命值为2，每负一场或平一场，生命值减1。当玩家生命值为0时，游戏结束。
注：在昆特牌原版规则中，平局时双方均损失生命值，只有尼弗迦德势力除外。故常规玩家一胜一平也可获得最终胜利。

- 决策类型

每个决策类型 (Decision Type) 都有自己对应的决策器 (Machine)，决策器通过各种算法来决定下一步的出牌，并且我们假设，一位玩家在一场游戏当中不会更改其策略类型。由于我们缺乏大量人类玩家的数据，需要用计算机蒙特卡洛模拟 (Monte Carlo Simulation) 出一定数量的对局作为学习的训练样本。对此，我们默认的决策类型是随机决策 (Random Decision)。在后面的学习中，我们逐步考虑各种模型，如随机梯度下降逻辑回归模型 (Stochastic Gradient Descent Logistic Regression) ，多层感知器神经网络模型 (Multi-Layer Perceptron) 和随机森林模型 (Random Forest) 等。有兴趣的读者，也可以以此为接口，加入人工决策系统，实现机器与真人的对战。

```
class Player:
    empCount = 0 # 总玩家数
    
    # 构造函数，用来定义对象
    def __init__(self, name):
        self.name = name
        self.deck = Board("Deck")            # 牌组
        self.hand = Board("Hand")            # 手牌
        self.deploy = Board("Deploy")        # 部署
        self.graveyard = Board("Graveyard")  # 弃牌堆
        self.passSignal = False              # 弃权标记
        self.health = 2                      # 生命值
        self.decisionType = 'Random'         # 决策类型，默认为随机决策
        self.machine = None                  # 决策器
        
    def setMachine(self, machine):
        self.machine = copy.deepcopy(machine)

    def setDeck(self, Deck = Board("Deck")):
        self.deck = Deck
        
    def setHand(self, Hand = Board("Hand")):
        self.hand = Hand
        
    def setDeploy(self, Deploy = Board("Deploy")):
        self.deploy = Deploy
        
    def setGraveyard(self, Graveyard = Board("Graveyard")):
        self.graveyard = Graveyard

    def setPass(self, signal = 0):
        self.passSignal = signal

    # 展示
    def display(self):
        print("==== Player Information ====")
        print("Player Name: ", self.name)
        print("Player Health: ", self.health)
        print("Player Pass Signal: ", self.passSignal)
        self.deploy.display()
        self.hand.display()
        self.graveyard.display()
        self.deck.display()
        print("==== End Player Information ====\n\n")
```

###### 玩家的基础操作

在这里，我们将考虑玩家的以下基本操作，这些操作独立于玩家的策略，是参与游戏的流程性操作。这里我们按照实现方式由简单到复杂依次排列。

```
class Player:
    # Continued

    # 损失生命值
    def lossHealth(self):
        self.health -= 1

    # 摸牌
    def drawCard(self, card):
        if self.deck.out(card):
            self.hand.add(card)
            return True
        return False   
        
    # 出牌
    def showCard(self, card):
        if self.hand.out(card):
            self.deploy.add(card)
            return True
        return False
    
    # 清理场地
    def clearDeploy(self):
        for card in self.deploy.cards.copy():
            self.deploy.out(card)
            self.graveyard.add(card)    

    # 弃权并跳过回合
    def passTurn(self, curRound = None, opponent = None):
        self.passSignal = True

    # 重置牌组
    def resetCards(self):
        self.passSignal = False
        self.health = 2
        for card in self.graveyard.cards.copy():
            self.graveyard.out(card)
            self.deck.add(card)
        for card in self.deploy.cards.copy():
            self.deploy.out(card)
            self.deck.add(card)
        for card in self.hand.cards.copy():
            self.hand.out(card)
            self.deck.add(card)
         
```


###### 玩家决策
在基础操作之上是策略操作，在这里，我们主要考虑三大类策略类型：
- 弃权

弃权 (Pass) 是最简单的策略类型，即一张牌不出，直接放弃比赛。当然并不会有理性的玩家选择这种策略，不过为了游戏设计的全面性，我们不应缺乏该策略的代码实现。

- 随机决策

随机决策 (Random Decision) 是另外一种很简单的策略类型。随机策略将成为我们训练初代机器人的主要样本来源。在这里我们考虑如下的实现方式：例如玩家有10张手牌，将玩家的手牌编号为0-9，并生成一个取值范围为0-10的整数型均匀分布随机数X，如果X<10, 则打出对应编号的手牌，否则，当前回合弃权。另外，如果当前没有手牌，则直接弃权。

- 机器学习策略

由于我们有多种机器学习的策略 (Strategy)，具体策略取决于玩家的决策类型和决策器，我们将其合并为一个函数。因此，我们需要对机器学习算法能够实现的功能做一个预期：机器学习究竟如何进行决策？

在这里，我们需要对游戏和决策进行一定的抽象层面的分析：什么是游戏？什么是决策？

这是一个很值得探讨的问题。由于我并没有人可以交换观点，暂时只好把自己的看法陈列如下：

策略 (Strategy) 是一系列决策 (Decision) 的集合，而游戏 (Game) 则是一些有确定数值评价的策略的集合。游戏中参与的双方，无非就是按照自己的策略交替地执行出牌的决策，最后我们在两种策略中选出一个，赋予其胜利者的称号，选出另一个，给与失败者的待遇。从数学的角度来说，我们将按照给定规则获胜的一方赋值为1，失败的一方赋值为-1，如果双方以平局收场 (前两局各胜一局，第三局平局)，那么双方赋值均为0。这样，对于每一次游戏，我们都可以有明确的标的，以 1/0/-1 作为输出，可以很方便地构建出有监督的学习机 (Supervised Learning Machine) 。

既然我定义的游戏是一系列决策的集合，那么应该对决策也有一个明确的定义。

在这里，我们认为，决策 (Decision) 是状态 (Status) 的一步转移。我们将各个状态考虑为海洋上有桥梁链接的诸多小岛，想要从陆路通过这一片海洋，我们只能由一个小岛出发，沿桥梁前往另一个小岛，而在每个小岛上选择哪座桥梁作为下一步的前进方向，便是我们的决策。我们会利用机器计算出来每一个小岛通向最终胜利和最终失败的概率，并根据概率给每个小岛一个相应的评分，评分越高的小岛，向它前进的价值越大。因此，我们的决策，将会采取较为贪心的算法，在每一个小岛上，找到和它相邻的评分最高的小岛，并选择它作为我们下一步的落脚点。这样，一步接一步地，我们将近似地以最大概率通向胜利。

```
class Player:
    # Continued

    # 决策
    def decision(self, curRound, opponent):
        if self.decisionType == "Pass":
            self.passTurn()
            return
        print(self.decisionType)
        self.strategy(curRound, opponent)
        

    # 随机决策
    def randomDecision(self, curRound = None, opponent = None):
        if self.hand.count == 0 or self.passSignal == True:
            self.passTurn()
        else:
            print(self.name + " Decision")
            cardIndex = random.randint(0, self.hand.count)
            if self.hand.count == cardIndex:
                print('Pass')
                self.passTurn()
            else:
                self.showCard(self.hand.cards[cardIndex])
                
    # 策略
    def strategy(self, curRound = None, opponent = None):
        if self.machine == None:
            self.randomDecision(curRound, opponent)
            return
            
        if self.hand.count == 0 or self.passSignal == True:
            self.passTurn()
        else:
            #存储所有的可能操作
            curStatusList = []
            playerCopy = copy.deepcopy(self)
            playerCopy.passTurn()
            curStatus = Status(playerCopy, opponent, curRound.num, curRound.turn, playerCopy)
            curStatusList = curStatusList.append(curStatus)
            curStatus.store()
            curDataFrame = curStatus.df
            
            for card in self.hand.cards:
                playerCopy = copy.deepcopy(self)
                playerCopy.showCard(card)
                curStatus = Status(playerCopy, opponent, curRound.num, curRound.turn, playerCopy)
                curStatus.store()
                curDataFrame = curDataFrame.append(curStatus.df)
                
            #print(curDataFrame)
            inputX = np.array(curDataFrame.drop(columns = ['WinSignal']))
            cardScore = self.machine.predict_proba(inputX)[:,2] - self.machine.predict_proba(inputX)[:,0]
            
            #print(cardScore)
            #选取所有可能操作中得分最高的执行
            showIndex = np.argmax(cardScore)
            #print(showIndex)
            
            if showIndex == 0:
                self.passTurn()
            else:
                self.showCard(self.hand.cards[showIndex - 1])

```



##### 4. 状态

聪明的读者已经发现，在上文中，我们为了介绍决策居然又引入了一个新的概念：状态 (Status)。那什么是状态呢？在这里，我们认为，理想情况下，状态应该是当前所有信息的合集。

我们所有的决策，都是基于当前状态的，并且只基于当前状态，因为我们所能依赖的历史上的所有信息，都必须以一定的方式存储于当下，才能为我们所接受所利用。举个例子来说，如果我们要研究生物进化论，研究恐龙的生活习性，只能通过留存下来的化石进行推测，猜想出来其体型和大致的生活习性。至于那些没有留存化石的恐龙，他们有多少只，他们喜欢吃什么，他们每天作息时间是什么样子的？因为缺乏了数据，便无法成为我们推理的依据。当然，随着技术的提高，我们在当下搜集的证据越来越多，可以通过各种技术手段，追溯历史上发生的事情，以丰富我们的认知，而不用真实地穿越回远古时期，设身处地的搜集当时的信息。

这告诉我们，只要当下状态存储的信息足够多，我们对未来的判断，可以不依赖于过去，而仅仅取决于当下。这种状态的性质叫做马尔科夫性 (Markovian)，是苏联著名数学家马尔科夫在研究概率论和随机过程时提出的。现实世界中，这样的模型是比较少见的，但是作为一个人工设计的游戏，我们实际上是可以实现的。

不过，考虑到状态中的很多信息除了占用我们的内存和算力，并不会对机器学习有很大的帮助，过多的信息可能反倒会将机器引入歧途。在学习的时候，我们可以考虑损失部分的马尔科夫性，只存储那些对游戏发展重要的变量，通过学习这些变量，达到窥一斑而见全豹的效果。在统计学中，这种变量有一个常用的名称，叫充分统计量 (Sufficient Statistics)，在这里我们就把它直接借用过来，即我们希望使用尽可能少的变量，来存储尽可能多的主要信息。本模型考虑了如下变量，他们将作为我们的近似“充分统计量”，共同构成“状态”这一类。

- 当前视角 (Perspective)
- 对手 (Opponent)
- 当前局数 (Round)
- 当前回合数 (Turn)
- 当前行动玩家 (Actor)
- 己方生命值 (MyHealth)
- 对方生命值 (OppoHealth)
- 己方弃权标记 (MyFlag)
- 对方弃权标记 (OppoFlag)
- 己方场上部署 (MyDeploy)
- 对方场上部署 (OppoDeploy)
- 己方手牌 (MyHands)
- 对方手牌数量 (OppoHandsNum)

- 获胜方 (Win)

在简化过的模型中，我们只考虑双方的场上牌数，场上总战斗力，手牌数和手牌总战斗力作为对场上和手牌信息的汇总。值得注意的是，场上信息是双方共享的，但是己方视角只能看到对方的手牌数，并不能看到对方的手牌力量总和 (尽管在当前设定下，可以通过过去的出牌记录推测出对方手牌的具体情形，我们还是隐藏了这部分信息，因为在真实对战中，我们无法知道对方的牌组和手牌)。也是在这种设定下，对于同一个回合，我们需要定义己方和对方两个视角，作为两条独立的状态，存储进我们的数据集 (Data Frame)。

```
class Status:
    # 构造函数，用来定义对象
    def __init__(self, Perspective, Opponent, Round, Turn, Actor):
        
        self.Perspective = Perspective.name # Perspective Name
        self.Opponent = Opponent.name # Perspective Name
        self.Round = Round # Round Num
        self.Turn = Turn # Turn Num
        self.Actor = Actor.name # Actor Name (Who Just Ends Action)
        self.ActorSignal = (Actor.name == Perspective.name)
        
        self.MyHealth = Perspective.health # Health Points
        self.OppoHealth = Opponent.health # Health Points
        
        self.MyFlag = Perspective.passSignal # Pass Flag T/F
        self.OppoFlag = Opponent.passSignal # Pass Flag T/F
        
        self.MyDeploy = copy.deepcopy(Perspective.deploy) # Deploy List
        self.OppoDeploy =  copy.deepcopy(Opponent.deploy)  # Deploy List
        
        self.MyHands =  copy.deepcopy(Perspective.hand)  # Hands List
        self.OppoHandsNum = Opponent.hand.count # Hands Num

        self.Win = 0
        
    # 胜利标记
    def win(self, player):
        if self.Perspective == player.name:
            self.Win = 1
        else:
            self.Win = -1
    
    # 展示
    def display(self):
        print("==== Status Information ====")
        print("Perspective: ", self.Perspective)
        print("Round: ", self.Round)
        print("Turn: ", self.Turn)
        print("Actor: ", self.Actor)
        print("Actor Signal", self.ActorSignal)
        print("My Health: ", self.MyHealth)
        print("Opponent Health: ", self.OppoHealth)
        print("My Flag: ", self.MyFlag)
        print("Opponent Flag: ", self.OppoFlag)
        print("My Deploy:")
        self.MyDeploy.display()
        print("Opponent Deploy:")
        self.OppoDeploy.display()
        print("My Hands:")
        self.MyHands.display()
        print("Opponent Hands Num:", self.OppoHandsNum)
        print("Win Signal: ", self.Win)
        print("==== Ends Status ====")
        print()
        
    # 存储，将当前状态存储为 DataFrame 的一行
    def store(self):
        self.dict = {'ActorSignal' : [self.ActorSignal], 
                     'Round' : [self.Round], 
                     'Turn' : [self.Turn],
                     'MyHealth': [self.MyHealth],
                     'OppoHealth': [self.OppoHealth],
                     'MyFlag': [self.MyFlag],
                     'OppoFlag': [self.OppoFlag],
                     'MyDeployCount': [self.MyDeploy.count],
                     'MyDeployPower': [self.MyDeploy.power],
                     'OppoDeployCount': [self.OppoDeploy.count],
                     'OppoDepolyPower': [self.OppoDeploy.power],
                     'MyHandsCount': [self.MyHands.count],
                     'MyHandsPower': [self.MyHands.power],
                     'OppoHandsCount': [self.OppoHandsNum],
                     'WinSignal': [self.Win]
                    }
        self.df = pd.DataFrame.from_dict(self.dict)
```



##### 5. 游戏
万事俱备，只欠东风。我们已经定义好游戏的各个要素，现在只需要再定义一个框架把他们整合起来，就可以进行游戏了。

大家如果还记得的话，在我们的规则中，一场游戏 (Game) 由2或3局 (Round) 组成，一局游戏由若干回合 (Turn) 组成，每回合双方各需打出一张手牌。因此，我们当然可以按照回合、局、游戏的顺序依次定义出三个对象，以完成游戏框架。不过，在编程的过程中，我发现，回合作为一个基础单位过于简单，似乎没有必要为一个回合单独定义一个对象，而且每个回合之间的联系较为密切，不甚容易分割。因此，在最终的实现中，我决定以局 (Round) 为基础对象进行设计，并在下面展开较为详细的叙述。

###### 局 (Round)
我们考虑一局游戏有以下主要属性：
- 计数
- 回合数
- 本局获胜玩家
- 双方弃权标记
- 状态列表

每一局游戏由若干回合构成，当每一个回合结束后，如果存在某一名玩家尚未弃权，则进行下一个回合。每一回合的操作又由**弃权判断**、**玩家决策**和**状态存储**三个部分组成。将他们顺次编写成函数，我们就得到了完整的“局”类。

```
class Round:

    # 构造函数，用来定义对象
    def __init__(self, roundNum, player1, player2):
        self.num = roundNum
        self.turn = 0
        self.winner = 0
        player1.passSignal = False
        player2.passSignal = False
        self.statusList = []
       
        
    # 执行回合
    def runTurn(self, player1, player2, displayIndex = False):
        if not player1.passSignal:
            player1.decision(self, player2)
            #self.displayRound(player1, player2)
            S11 = Status(player1, player2, self.num, self.turn, player1)
            self.statusList.append(S11)
            
            S12 = Status(player2, player1, self.num, self.turn, player1)
            self.statusList.append(S12)
            
            if displayIndex:
                S11.display()
                S12.display()
            
        if not player2.passSignal:
            player2.decision(self, player1)
            #self.displayRound(player1, player2)
            S21 = Status(player1, player2, self.num, self.turn, player2)
            self.statusList.append(S21)
            
            S22 = Status(player2, player1, self.num, self.turn, player2)
            self.statusList.append(S22)
            
            if displayIndex:
                S21.display()
                S22.display()
            
            
    # 判断是否执行下一回合
    def checkTurn(self, player1, player2):
        if player1.passSignal and player2.passSignal:
            return False
        else:
            return True
    

    # 判断本局游戏获胜者
    def checkRoundWinner(self, player1, player2):
        if player1.deploy.power > player2.deploy.power:
            self.winner = 1
            player2.health -= 1
            print(player1.name, " Wins Round ", self.num)
        else:
            if player2.deploy.power > player1.deploy.power:
                self.winner = -1
                player1.health -= 1
                print(player2.name, " Wins Round ", self.num)
            else:
                self.winner = 0
                player1.health -= 1
                player2.health -= 1
                print("Tie Round ", self.num)          
        player1.clearDeploy()
        player2.clearDeploy()
    

    # 执行局
    def runRound(self, player1, player2, displayIndex = False):
        print("Round", self.num)
        while self.checkTurn(player1, player2):
            self.turn = self.turn + 1
            print("Turn ", self.turn)
            self.runTurn(player1, player2, displayIndex)
        self.checkRoundWinner(player1, player2)


    # 展示
    def displayRound(self, player1, player2):
        player1.display()
        player2.display()
```

###### 游戏 (Game)

如法炮制，将回合之于局转化为局之于游戏，我们就得到了一场游戏的代码。我们可以看到，本来设计一个游戏是一件极其繁琐的事情，但是，如果我们把它分解成一个个子对象，并将每一个子类完整清晰地定义出来，最后实现“游戏”的代码反而是各个类中几乎最为简单的。将程序的各个组成部分模块化，这便是面向对象编程的优势之一。

```
class Game:

     # 构造函数，用来定义对象
    def __init__(self, player1, player2):
        self.round = 0
        self.winner = 0
        player1.resetCards()
        player2.resetCards()
        self.statusList = []
        

    # 判断游戏的获胜者
    def checkGameWinner(self, player1, player2):
        if player1.health > player2.health:
            self.winner = 1
            print(player1.name, " Wins the Game!")
            for curStatus in self.statusList:
                curStatus.win(player1)
        else:
            if player2.health > player1.health:
                self.winner = -1
                print(player2.name, " Wins the Game!")
                for curStatus in self.statusList:
                    curStatus.win(player2)
            else:
                self.winner = 0
                print("Ties!")
            

    # 执行游戏
    def runGame(self, player1, player2, displayIndex = False):
        for card in player1.deck.cards.copy():
            if displayIndex:
                card.displayCard()
            player1.drawCard(card)
        for card in player2.deck.cards.copy():
            if displayIndex:
                card.displayCard()
            player2.drawCard(card)
            
        while player1.health > 0 and player2.health > 0:
            self.round = self.round + 1
            curRound = Round(self.round, player1, player2)
            curRound.runRound(player1, player2, displayIndex)
            if displayIndex:
                curRound.displayRound(player1, player2)
            self.statusList = self.statusList + curRound.statusList
            
        self.checkGameWinner(player1, player2)
        
```

### 游戏流程与数据整理
到目前为止，我们已经设计好游戏中所有的类和对象，现在只需要调用上述的各个函数，即可完成游戏的流程。我们将实现游戏的代码展示如下。由于缺乏大量的人类玩家的数据，第一代的玩家对象将会采用随机决策的方式完成比赛。
```
player1 = Player('Player1')
player2 = Player('Player2')

for card in cardPool.copy():
    player1.hand.add(card)
    player2.hand.add(card)

game1 = Game(player1, player2)
game1.runGame(player1, player2, displayIndex = True)
```
我们不妨来看一局随机决策的选手们的对局：
```
=========================================
Round 1
-----------------------------------------
Player1 Deploy: Bishop        Power:  7
Player2 Deploy: Knight        Power:  4
Player1 Deploy: Catapult      Power:  8
Player2 Deploy: Bishop        Power:  7
Player1 Deploy: Geralt        Power:  10
Player2 Deploy: Triss         Power:  9
Player1 Pass
Player2 Deploy: Rook          Power:  6
Player2 Deploy: Elite         Power:  5
Player2 Deploy: Infantry      Power:  3
Player2 Deploy: Catapult      Power:  8
Player2 Deploy: Geralt        Power:  10
Player2 Deploy: Guard         Power:  2
Player2 Pass
-----------------------------------------
Player1 Total Power: 25
Player2 Total Power: 54
Player2  Wins Round  1

=========================================
Round 2
-----------------------------------------
Player1 Deploy: Elite         Power:  5
Player2 Pass
Player1 Deploy: Triss         Power:  9
Player1 Deploy: Infantry      Power:  3
Player1 Deploy: Knight        Power:  4
Player1 Deploy: Pawn          Power:  1
Player1 Pass
-----------------------------------------
Player1 Total Power: 22
Player2 Total Power: 0
Player1  Wins Round  2

=========================================
Round 3
-----------------------------------------
Player1 Deploy: Guard         Power:  2
Player2 Deploy: Pawn          Power:  1
Player1 Pass
Player2 Pass
-----------------------------------------
Player1 Total Power: 2
Player2 Total Power: 1
Player1  Wins Round  3

=========================================
Player1  Wins the Game!
```

我们可以发现，和我预期的一样，随机的决策的玩家，宛如刚刚接触游戏的初学者，仅仅能完成比赛，而打不出任何的章法。比如玩家二的操作，在第一局对方已经放弃的情况下，自己还在不知收敛地肆意出牌，几乎将所有的手牌全部打光，导致第二局和第三局的惨败。

接下来，我们将以次为基础，蒙特卡洛 (Monte Carlo) 生成大量的比赛数据，并在这些比赛数据中寻找规律，训练出能够取胜的人工智能 (Artificial Intelligent) 玩家。我们将所有的比赛状态数据存储为DataFrame格式，定义并执行重复比赛的函数，并将数据存放在.csv文件中：
```
def repeatGame(player1, player2, num = 100):
    GameHistory = []
    
    GameDatabase = pd.DataFrame(columns=('ActorSignal', 
                     'Round', 
                     'Turn',
                     'MyHealth',
                     'OppoHealth',
                     'MyFlag',
                     'OppoFlag',
                     'MyDeployCount',
                     'MyDeployPower',
                     'OppoDeployCount',
                     'OppoDepolyPower',
                     'MyHandsCount',
                     'MyHandsPower',
                     'OppoHandsCount',
                     'WinSignal'))
    
    
    for i in range(0,num):
        game = Game(player1, player2)
        game.runGame(player1, player2)
        GameHistory.append(game.winner)
        for S in game.statusList:
            S.store()
            GameDatabase = GameDatabase.append(S.df, ignore_index = True)
        
        
    print(player1.name + " first.")
    print(player1.name + " wins " + str(GameHistory.count(1)) + " times.")
    print(player2.name + " wins " + str(GameHistory.count(-1)) + " times.")
    print("Tie " + str(GameHistory.count(0)) + " times.")
    
    return GameDatabase

GameDatabase = repeatGame(player1, player2)

GameDatabase.to_csv('GameDatabase.csv', mode='a', index=False, header=False)
```

重复多次的游戏，我们就得到了一个有足够训练样本的数据集：

| No. | Actor Signal | Round | Turn | My Health | Oppo Health | My Flag | Oppo Flag | My Deploy Count | My Deploy Power | Oppo Deploy Count | Oppo Depoly Power | My Hands Count | My Hands Power | Oppo Hands Count | Win Signal |
|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:| :----:| :----:| :----:|
| 0 | True | 1 |	1 |	2 | 2 | True | False | 0 | 0 | 0 | 0 | 10 | 55 | 10 | -1 |
| 1 | False | 1 | 1 | 2 | 2 | False | True | 0 | 0 | 0 | 0 | 10 | 55 | 10 | 1 | 
| 2 | False | 1 | 1 | 2 | 2 | True | False | 0 | 0 | 1 | 6 | 10 | 55 | 9 | -1 | 
| 3 | True | 1 | 1 | 2 | 2 | False | True | 1 | 6 | 0 | 0 | 9 | 49 | 10 | 1 | 
| 4 | False | 1 | 2 | 2 | 2 | True | False | 0 | 0 | 2 | 9 | 10 | 55 | 8 | -1  | 
| ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | 

有了这些训练集，机器学习的算法就可以派上用场了。

### 机器学习

何谓机器学习 (Machine Learning)？简而言之，就是给机器以一定的数据，让它从这些数据中寻找规律，从而达到可以对新的输入数据给出预测和反馈的过程。整个过程对已有经验的总结，得到规律，并将规律应用到新的情景中，极其类似于人类的学习过程，因此我们称之为机器学习。

从这个角度来说，只要能满足给定输入、发现规律、给出输出，机器学习的样式可简单可复杂，最简单的机器学习有线性回归 (Linear Regression)，复杂的机器学习则到现在依旧是科研的最新话题。在这里，我们将尝试以下几种机器学习的模型，并用实战来检验他们的效果。

##### 1.逻辑回归
逻辑回归 (Logistic Regression) ，又译为逻辑斯蒂回归，当然不管是哪个音译对我们理解都完全没有帮助。我猜想，Logistic 一词在这里应为“符号化表示”的意思，亦即我们的回归目标是一个符号类型的变量，而不是传统意义上的数值型变量。这很符合我们的需求：我们的目标输出变量是当前状态是否赢得最终胜利的标记 (Win Signal)，而获胜标记只有胜平负三种类型，而不存在例如0.87胜，0.13负这种中间的数值变量。这也是我们选择逻辑回归而不是最常用的线性回归的原因。

我们现在来阐述逻辑回归模型，为了方便起见，我们可以先简化一下模型，假设只有胜 (1) 和不胜 (0) 两种状态。假设我们的数据集中共有 n 个不同的状态，其中第 $ i $ 个状态 $ (i = 1, ..., n) $ 的各项信息的值记为 $ X_i $, $ X_i $是一个多维向量，其分量包括回合数、生命值等，$ X_1, ... X_n $将作为我们模型的输入。同时，第 $ i $ 个状态的胜负标记为 $ Y_i $, $Y_i \in \{ 0, 1\} $ 将作为我们的输出变量。在逻辑回归模型中，我们假设各个状态的胜负 $Y_i$ 是服从于某个伯努利分布 (Bernoulli Distribution) 的随机变量 (Random Variable)，其获胜的概率 $ p_i $ 则由某个与 $X_i$ 有关的逻辑斯蒂方程决定，其表达式为

$$
p_i = \dfrac{e^{ X_i^T \beta } }{ 1 + e^{ X_i^T \beta } }.
$$

其中，$ \beta $ 是模型假定的一组参数，我们假定在所有状态里，这一组参数不发生变化。于是他们便成了所有状态都需要遵循的规律，也是我们机器学习的目标所在。逻辑斯蒂函数的图像如下所示：

![逻辑斯蒂方程 (Logistic Function)](https://upload-images.jianshu.io/upload_images/13641295-4a5a8ccd20c3eba6.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

我们可以观察到，逻辑斯蒂曲线随 $X^T \beta$ 单调上升，当 $ X^T \beta$ 趋向于负无穷时，获胜概率 $ p_i $ 趋向于0，当 $ X_i^T \beta$ 趋向于正无穷时，获胜概率 $ p_i $ 趋向于1，当 $X_i^T \beta = 0$ 时，获胜概率 $p_i = 0.5$ 。因此，我们可以通过样本数据调节参数 $\beta$ 的大小，使获胜状态下 $X_i^T \beta$ 偏大，而失败状态下反之。由此得到一组拟合出来的系数 $\beta$，将之应用在下一步的选择 $X_{next}$ 中，由于游戏的本质规律没有变化，我们仍然可以得到若 $X_{next}^T \beta$ 越大，则选择该步后的获胜概率越大。这也就达到了帮助我们决策的目的。

明确我们要调节的参数后，另外一个问题便是如何优化之。想要优化，首先要给定评价标准，什么样的模型我们可以称得上是好的模型。通常情况下，我们可以认为，样本总是倾向于出现在给定参数下最可能出现的状态。因此我们可以定义其出现的可能性为似然性 (Likelihood)，并且寻找一组使得其似然性最大的参数 $\hat \beta$，作为我们对参数 $\beta$ 的估计。这种参数估计的办法被称为极大似然法 (Maximum Likelihood Estimation) 。在伯努利分布的假设下，样本的对数似然函数 (Log-Likelihood Function) 为

$$
l(\beta) = \sum_{i=1}^{N} (Y_i \log p_i + (1-Y_i) \log (1-p_i) )
=\sum_{i=1}^{N} (Y_i X_i\beta^T  - \log (1+ e^{X_i^T \beta}) ).
$$

我们所要做的是，寻找一组参数 $\beta$， 使得上述 $l(\beta)$ 最大。考虑到在优化问题中，通常以求解极小值问题为目标，将 $ l(\beta) $ 最大化亦等价于将 $-l(\beta)$ 最小化，我们可以考虑一个一般意义上的最小值求解算法，并将上述问题带入，便可求得最优参数 $\hat \beta$ ，即

$$
\hat \beta = \arg \max_{\beta} l(\beta).
$$

###### 随机梯度下降的最优值求解算法

我们现在将问题简化处理，假设在一个四面环山的盆地中，我们位于一侧的山坡上，想要到达盆地的最低点，我们应当往哪个方向走呢？直觉告诉我们，最快的方法一定是沿着山坡最陡峭的方向向下。从数学的角度上来说，我们将山坡最陡峭的方向称之为梯度 (Gradient)，沿着梯度方向走到该方向的最低点，然后再次寻找梯度方向继续前进，直到收敛为止，这一套算法称之为梯度下降法 (Gradient Descent Method)。尽管它不是一个最优化的办法，但是由于其简单易行且易于理解，成为了优化设计中最常用的方法之一。

不过在机器学习领域，使用梯度下降法常常会遇到一个困难：由于样本数量巨大，一次性将所有的数据全部带入目标函数求解梯度在计算量上过于庞大，超出了一般计算机的承受范围。于是，人们开始构想，能不能将巨大的训练数据集分割成为一个个较小的数据块，每次喂给目标函数一个小块数据，既不超过它的承载极限，又能够得到最优解呢。事实上，有一个折中的方案，即把数据集分割成小块逐次训练的方法被称之为批梯度下降算法 (Mini-batch Gradient Descent)。而为了保证得到的最优值结果与原算法保持一致，我们需要将原数据集随机地分成等大小的小块，这种随机分割训练数据集的算法，又被称为随机梯度下降法 (Stochastic Gradient Descent)。随机梯度下降既解决了原数据集过大无法操作的问题，又保证了得到最优解的期望收敛于真实最优值，同时我们可以通过控制分块数据的大小来保证学习结果的稳定性。因此，随机梯度下降的算法如今被广泛应用在机器学习算法的优化领域。

我们可以将随机梯度下降法的算法框架描述如下：
1. 输入：初始参数估计 $\beta_0$，置 $k = 0$；
2. 若 $\beta_k$ 不满足收敛条件，进行步骤3；否则，进行步骤6；
3. 选择步长 $\gamma_k$ ，将数据随机均匀分为 $B_k$ 块，记为 $U_1, ..., U_{B_k}$，计算各块梯度$G(\beta_k, U_b)$，取其平均值
$$
G_k =\dfrac{1}{B_k} \sum_{b=1}{B_k} G(\beta_k, U_b),
$$
并将 $-G_k$ 作为当前下降方向；
4. 沿当前下降方向前进，前进步长为
$$\beta_{k+1} = \beta_{k} - \gamma_k G_k;$$
5. 置 $k = k+1$，并返回步骤2；
6. 输出： 最优参数$\beta_k$ 。

结合以上两个模块，我们得到了一个完整的构建模型的方法，但是由于其细节过于繁琐，在实践中我们使用了Python包sk-learn中自带的随机梯度模型，其具体代码实现如下：

```
# 读取训练数据集
reader_init = pd.read_csv("GameDatabase.csv", header = 0)

# 随机梯度下降，逻辑回归学习器
sgd_clf = linear_model.SGDClassifier(loss="log", max_iter=1000, warm_start=True)

# 训练输入
X = np.array(reader_init.drop(columns = ['WinSignal']))

# 训练输出
y = np.array(reader_init['WinSignal'])

# 训练模型
sgd_clf.fit(X, y)
```

###### 训练效果检测

Sk-learn 将随机梯度和逻辑回归进行了很好的封装，因此，上述几行代码就完成了模型的训练。为了检测模型训练效果如何，我们将训练好的随机梯度逻辑回归策略赋予新的玩家，并让新玩家与随机策略玩家对战。

```
# 随机梯度逻辑回归玩家
player3 = Player('Player3')
player3.decisionType = "SGDStrategy"
player3.setMachine(sgd_clf)

# 随机策略玩家
player4 = Player('Player4')
player4.decisionType = "Random"

for card in cardPool.copy():
    player3.hand.add(card)
    player4.hand.add(card)

game = Game(player3, player4)
game.runGame(player3, player4, displayIndex = True)
```
其对局结果如下：
```
=========================================
Round 1
-----------------------------------------
Player3 Deploy: Geralt        Power:  10
Player4 Deploy: Bishop        Power:  7
Player3 Deploy: Triss         Power:  9
Player4 Deploy: Pawn          Power:  1
Player3 Deploy: Catapult      Power:  8
Player4 Deploy: Geralt        Power:  10
Player3 Deploy: Bishop        Power: 7
Player4 Deploy: Triss         Power:  9
Player3 Deploy: Rook          Power:  6
Player4 Deploy: Guard         Power:  2
Player3 Deploy: Elite         Power:  5
Player4 Deploy: Knight        Power:  4
Player3 Deploy: Knight        Power:  4
Player4 Deploy: Catapult      Power:  8
Player3 Deploy: Infantry      Power:  3
Player4 Deploy: Rook          Power:  6
Player3 Deploy: Guard         Power:  2
Player4 Deploy: Elite         Power:  5
Player3 Deploy: Pawn          Power:  1
Player4 Pass
Player3 Pass
-----------------------------------------
Player3 Total Power: 55
Player4 Total Power: 52
Player3  Wins Round  1

=========================================
Round 2
-----------------------------------------
Player3 Pass
Player4 Deploy: Infantry      Power:  3
Player4 Pass
-----------------------------------------
Player3 Total Power: 0
Player4 Total Power: 4
Player4  Wins Round  2

=========================================
Round 3
-----------------------------------------
Player3 Pass
Player4 Pass
-----------------------------------------
Player3 Total Power: 0
Player4 Total Power: 0
Round  3 Ties.

=========================================
Ties!
```
从对局中我们可以看到，装备机器的玩家三确实展现出了其独特的策略，从大到小依次出牌。在第一局中，两玩家打的如火如荼，玩家三手牌倾囊而出，玩家四则有所保留最后存了一张手牌到第二局。由于战事焦灼，最终双方各赢一局，平手收场。

为了消除比赛中的偶然因素，我们决定让两玩家重复比赛100局，以观察装配机器学习策略的玩家三是否有稳定的发挥，以及机器学习策略能否能真正战胜毫无规律的随机出牌策略。为此，我们定义了如下函数：

```
def repeatGame(player1, player2, num = 100):
    GameHistory = []
    
    GameDatabase = pd.DataFrame(columns=('ActorSignal', 
                     'Round', 
                     'Turn',
                     'MyHealth',
                     'OppoHealth',
                     'MyFlag',
                     'OppoFlag',
                     'MyDeployCount',
                     'MyDeployPower',
                     'OppoDeployCount',
                     'OppoDepolyPower',
                     'MyHandsCount',
                     'MyHandsPower',
                     'OppoHandsCount',
                     'WinSignal'))
    
    
    for i in range(0,num):
        game = Game(player1, player2)
        game.runGame(player1, player2)
        GameHistory.append(game.winner)
        for S in game.statusList:
            S.store()
            GameDatabase = GameDatabase.append(S.df, ignore_index = True)
        
        
    print(player1.name + " first.")
    print(player1.name + " wins " + str(GameHistory.count(1)) + " times.")
    print(player2.name + " wins " + str(GameHistory.count(-1)) + " times.")
    print("Tie " + str(GameHistory.count(0)) + " times.")
    
    return GameDatabase
```
我们将上述玩家三与玩家四的对局重复100次后，战况如下：
```
Player3 wins 31 times.
Player4 wins 37 times.
Tie 32 times.
```
双方几乎打成平手，甚至随机出牌的玩家四略胜一筹。为什么经过训练，有专业模型指导的玩家三，反而不如出牌毫无章法的玩家四呢？我们总结发现，原因出在了用于学习的机器上。在我们的模型中，尽管各个参数都会对我们的预测产生影响，但是在简单的线性假设下，各个变量的影响相互独立，难以形成一股合力，反而经常互为掣肘。例如，大量的训练样本告诉我们，场上战斗力越大，获胜的几率越大，于是机器在这一发现的指引下，不分场合拼命出牌，在第一局将所有的牌全部打出，导致后继乏力，无法获得最终胜利。理性的玩家则应该分析出来，当第一局游戏自己占优势后，再多出的牌对获胜不但无益，反倒有害，而第三局游戏，则应该将所有的手牌全部打出以求决战。在这一层面上，游戏的局数和场上的战斗力共同作用于最终的胜利，而当前的模型无法处理这一现象。

因此，我们在接下来换用更加复杂的模型来训练新机器。

##### 2. 神经网络
神经网络 (Neural Network) 这一术语在近几年来火遍大街小巷，它通过模拟人类神经元的信息传递机制，构造了数据神经元 (Neoron)，并用数据神经元之间的函数模拟神经元中的突触和信息传递。当数据神经元的个数足够多，数据传递的层数也足够多时，这些神经元便构成了一个人工神经网络 (Artificial Neural Network)或多层感知器 (Multi-Layer Perception)，我们认为，人工神经网络的反应可以模拟人脑的反应，从而感知和学习新的事物。

最简单的单层神经网络与前述的逻辑回归模型没有太大的不同，但是神经网络的优势是，叠合了多个不同的回归，从而使得最终的结果摆脱了简单线性的限制，能够模拟各种各样复杂的规则。我们以一个三层的简单网络为例来简要介绍神经网络的工作原理。一个三层的神经网络包括输入层 (Input)，隐藏层 (Hidden Layer) 和输出层 (Output)。隐藏层位于输入层与输出层之间，输入层的线性关系在这里得到非线性整合重组，再送往输出层做最终的预测。由于隐藏层的加入，各个输入元素之间不再是各自为战，从而增加了机器的感知范围和学习能力。神经网络预测模型输出的过程被称作向前传播 (Forward Propagation)，而我们通过预测输出与实际输出之间的误差，调节参数的过程称作向后传播 (Backward Propagation)。通过不断地向前向后传播，我们可以将各个输入的参数调节到一个最优或者近似最优的值，从而使得模型得到的输出尽可能地拟合实际的输出。

![神经网络示意图](https://upload-images.jianshu.io/upload_images/13641295-60c7f96896c539cd.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

由于神经网络的数学实现较为复杂，有兴趣的读者可参考机器学习专业教材，这里作者略去，仅展示调用并实现神经网络模型所需的代码。在这里，我们考虑一个有两个隐藏层的神经网络，每个隐藏层有100个神经元节点，并调用 sklearn.neural_network 包中的函数：

```
mlp_clf = MLPClassifier(solver='lbfgs', hidden_layer_sizes = (100,100), alpha=1e-5, random_state=1, warm_start=True)
mlp_clf.fit(X,y)

# 神经网络玩家
player5 = Player('Player5')
player5.decisionType = "MLPStrategy"
player5.setMachine(mlp_clf)

# 随机策略玩家
player6 = Player('Player6')
player6.decisionType = "Random"

for card in cardPool.copy():
    player5.hand.add(card)
    player6.hand.add(card)

game = Game(player5, player6)
game.runGame(player5, player6, displayIndex = True)
```
我们现将其对局结果展示如下：
```
=========================================
Round 1
-----------------------------------------
Player5 Deploy: Elite         Power:  5
Player6 Deploy: Knight        Power:  4
Player5 Deploy: Guard         Power:  2
Player6 Deploy: Geralt        Power:  10
Player5 Deploy: Geralt        Power:  10
Player6 Deploy: Triss         Power:  9
Player5 Deploy: Triss         Power:  9
Player6 Deploy: Guard         Power:  2
Player5 Deploy: Rook          Power:  6
Player6 Pass
Player5 Deploy: Knight        Power:  4
Player5 Pass

-----------------------------------------
Player5 Total Power: 36
Player6 Total Power: 25
Player5  Wins Round  1

=========================================
Round 2
-----------------------------------------
Player5 Deploy: Bishop        Power:  7
Player6 Deploy: Elite         Power:  5
Player5 Deploy: Catapult      Power:  8
Player6 Deploy: Rook          Power:  6
Player5 Pass
Player6 Deploy: Infantry      Power:  3
Player6 Pass
-----------------------------------------
Player5 Total Power: 15
Player6 Total Power: 14
Player5  Wins Round  2

=========================================
Player5  Wins the Game!
```
从对局结果中，我们发现，配备神经网络的玩家似乎已经有了一定的策略，例如在对方出高战斗力的牌 Geralt 和 Triss 时，自己也跟着出相同的牌，在对方停止出牌后，不再一味地高歌猛进，而是将第一局适时收官。这些都反映了神经网络对游戏规则的较好领会。然而，我们也可以发现，神经网络玩家的出牌策略仍然存在缺陷，例如在第一局对方停止出牌后，玩家五就已经稳操胜券了，却毫无必要地多出了一张 4 分的 Knight 牌。由此可见，当前的神经网络实际还有很大的优化的空间。

我们统计了100局神经网络玩家与随机策略玩家的对局，其结果如下：
```
Player5 wins 75 times.
Player6 wins 12 times.
Tie 13 times.
```
我们发现，尽管还存在缺陷，神经网络在对战初学者时已经能够具备压倒性的优势了。如果我们继续增加神经网络的训练样本或者神经网络的复杂性，便有可能得到战术更加完善的打牌机器人了。另外，考虑到当前的神经网络是从完全随机的对局中学习经验，沙里淘金，其学习效率必然会大打折扣。如果有高质量的对局作为输入，则神经网络的水平有可能会得到飞跃性的提升。对此，鉴于作者才疏学浅精力有限，本文内就不再深入讨论了。


##### 3. 随机森林
随机森林(Random Forest)是由多个决策树(Decision Tree)组合而成的集成学习模型(Ensemble Learning Model)。

决策树模型的工作原理是在数据的某些特征上插入一些节点，根据新输入数据在这些特征上与节点值的比较完成分类。例如，当我们希望辨别某个人的性别时，往往会将头发长度当做一个重要的特征，如果其头发长度超过15厘米，则我们倾向于认为此人是个女性，反之则认为是男性。不过，单纯以头发长度为标准分类过于粗犷，我们可以加入一些其他特征的判断标准，例如身高是否超过170厘米，嗓音是高音还是低音。如果一个观察对象虽然头发短于15厘米，但是身高低于170厘米，嗓音偏高音，那么我们仍然偏向于认为这是一位女性。由此，我们通过头发这一个特征将人群分为了两类，在每一类下再找到一些其他的特征，构造新的节点，将每一个子类进行二次分类，即可得到一个分叉的树状结构，每一个分支下的所有样本归入同一个类别。按照这个给定的规则进行分类的模型就叫做决策树了。下图展示的是一个用决策树模型来进行诈骗邮件分类的模型。

![决策树模型示例](https://upload-images.jianshu.io/upload_images/13641295-8a30210a9a8f4967.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

不过我们也发现，决策树模型有一个致命的缺点：当我们选取的节点个数过少时，容易出现大量误判，而当节点数量过多时，决策规则过于复杂，又可能得到一个过拟合的模型，即我们构造出来的决策树只对训练集里的数据有效，对未来数据做预测时效果就不那么理想了。为了解决这一问题，我们可以构造多个决策树，每个决策树只关注样本的部分特征，然后将多个决策树的分类结果进行集成，这样，虽然每一个决策树给出的结果都有一定可能的偏差，但是把所有决策树的结果综合起来，择其优者而从之，则可以得到一个更加稳定可靠的结果了。这就是民主的力量，也正契合了一句古话所言，偏信则暗，兼听则明。由于我们最终的决策器由众多的决策树构成，构建每棵决策树时会随机挑选样本的部分特征，因此这一类模型被形象地称为随机森林模型。随机森林模型的具体实现细节，感兴趣的读者可以参考机器学习相关书籍。在这里，我们直接调用sklearn.ensemble包中的RandomForestClassifier 模块实现随机森林的学习过程，具体调用代码如下：


```
# 随机森林模型，每棵随机数的最大深度设置为10
rf_clf = RandomForestClassifier(max_depth=10,  warm_start = True)

rf_clf.fit(X, y)
rf_clf.score(X, y)

# 随机森林玩家
player7 = Player('Player7')
player7.decisionType = "RFStrategy"
player7.setMachine(rf_clf)

# 随机决策玩家
player8 = Player('Player8')
player8.decisionType = "Random"

for card in cardPool.copy():
    player7.hand.add(card)
    player8.hand.add(card)

game = Game(player7, player8)
game.runGame(player7, player8, displayIndex = True)
```

我们再来看一下随机森林玩家的对阵情况：
```
=========================================
Round 1
-----------------------------------------
Player7 Deploy: Pawn          Power:  1
Player8 Deploy: Elite         Power:  5
Player7 Deploy: Catapult      Power:  8
Player8 Deploy: Guard         Power:  2
Player7 Deploy: Triss         Power:  9
Player8 Deploy: Knight        Power:  4
Player7 Pass
Player8 Deploy: Bishop        Power:  7
Player8 Deploy: Rook          Power:  6
Player8 Deploy: Triss         Power:  9
Player8 Deploy: Pawn          Power:  1
Player8 Deploy: Geralt        Power:  10
Player8 Pass

-----------------------------------------
Player7 Total Power: 18
Player8 Total Power: 44
Player8  Wins Round  1

=========================================
Round 2
-----------------------------------------
Player 7 Deploy: Infantry      Power:  3
Player 8 Deploy: Catapult      Power:  8
Player 7 Deploy: Geralt        Power:  10
Player 8 Pass
Player 7 Deploy: Bishop        Power:  7
Player 7 Pass

-----------------------------------------
Player7 Total Power: 20
Player8 Total Power: 8
Player7  Wins Round  2

=========================================
Round 3
-----------------------------------------
Player7 Deploy: Rook           Power:  6
Player8 Deploy: Infantry       Power:  3
Player7 Deploy: Elite          Power:  5
Player8 Pass
Player7 Deploy: Guard          Power:  2
Player7 Deploy: Knight         Power:  4
Player7 Pass

-----------------------------------------
Player7 Total Power: 17
Player8 Total Power: 3
Player7  Wins Round  3

=========================================
Player7  Wins the Game!
```
我们可以发现，对阵随机决策的玩家，机器学习加成下的随机森林玩家仍然轻松取胜。但是，我们在神经网络玩家身上发现的问题仍然存在，诸如在己方占优而对手已经弃权的情况下仍然继续出牌等等。再次统计100局随机森林玩家与随机策略玩家的对局，其结果如下：
```
Player7 wins 95 times.
Player8 wins 1 times.
Tie 4 times.
```
从对局结果来看，随机森林玩家似乎是比神经网络玩家有了大幅度的改进。其中一个很大可能的原因是，随机森林模型是基于样本特征的模型，不论特征是连续变化的随机变量(例如头发长度、身高等)还是离散型的随机变量(例如嗓音高低、头发颜色等)，我们都可以构建相应的随机决策树进行分类和决策。而神经网络终究还是一个连续的函数变化，处理既有连续型也有离散型变量的输入数据时，就需要改进其数据结构后再进行建模和预测，而这一部分的操作，则需要读者们更进一步的思考了。

### 总结和后记
我们从面向对象的程序开始讲起，逐步复现了昆特牌的基本操作，从底层探讨了一个卡牌类对战游戏的逻辑和设计。可以看到，只要我们将程序仔细分解，各个击破，设计一款游戏也并不是非常困难的事情。完成了底层设计之后，加入一些统计学的工具，我们就可以实现一些机器学习的算法，这些算法可以从历史数据中自动学习和改进，最终达到能够与人类对弈的水平，也就是逐步进化为人工智能了。当然，限于作者的水平和精力，本文中提出的很多问题并没有得到解决，例如：以上一代的机器学习的对弈结果作为学习样本训练出的新一代机器是否能够完全战胜上一代机器？再下一代机器是否又会全方位超越新一代的机器呢？而如果我们不断训练，是会得到一代代更加优秀的机器，还是在某一代之后，这种进化就止步不前了呢？再推而广之，是否存在一个昆特牌的最优打法，使得我们设计的机器学习打法在不断优化后逐渐接近甚至达到最优打法呢？这一系列的问题，我相信，即使在今天的最顶尖机器学习专家的研究领域，也是有其探讨的价值的。

另外，鉴于本文是作者临时起意随手而写的，实验设计挂一漏万，很多地方可能考虑欠妥，例如，神经网络的层数如何设计，对弈时的先后手如何转换和平衡等。如果有读者发现了代码中有明显的bug或者是能够优化的地方，万望能够百忙之中通知我，我一定会竭力进行修改和优化。如果有读者对我在文中的问题有研究的兴趣，我也愿意竭诚合作，尽我所能，以期有所学术突破。

最后，关于本文的创作，我在这里留下一点点感想。我是2021年春天，百无聊赖之中，在罗图的推荐下入手的《巫师3：狂猎》，也是在那个学期开始担任机器学习助教的。无意中发现兴许有可能用自己所学解决一下游戏中的问题，于是着手开始写代码。基础代码在2021年暑假基本写就，也写完了大部分的文本内容。当时与同学探讨，认为还可以把这个代码完善，并且深入探究一番，做一些学术课题，想出一大堆工作可以做。可随后冗事缠身，得有闲暇时又缺少了精神来做它，对于困扰的问题也想不到直接的解决办法，于是搁置良久。足见行百里者半九十，非虚言也。直至今年夏天，偶然记起还有这个半成品的文章，想到既然大的课题做不了，把去年写出的部分进行一下收尾，写成一篇科普性的小文章也很不错了。于是我花三两天草草了事，把去年的工作封装完善，简单地拼凑成这么一篇小短文，如此结束，当然不免也有些虎头蛇尾之嫌了。不过，至少这是一次对所学的简单实践，纵然并不一定有很多科学研究价值，但能够部分满足一下作者的兴趣，也不失写出它的意义了。

2022年5月25日于佐治亚

