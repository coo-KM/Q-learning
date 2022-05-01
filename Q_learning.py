import numpy as np
import matplotlib.pyplot as plt
import pygame
import sys

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (96, 96, 96)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
ORANGE = (255, 165, 0)
SCREEN_SIZE = 20 * 20

class Agent:
    def __init__(self,alpha=0.1,gamma=0.9,eps=0.1,acts=None,now_x=None,now_y=None):
        self.alpha=alpha
        self.gamma=gamma
        self.eps=eps
        self.reward = []
        self.acts = acts
        self.prv_act=None
        self.now_x=now_x
        self.prv_x = None
        self.now_y=now_y
        self.prv_y=None
        self.q_val=np.zeros((12,12,4))

    #ε-greedyにより行動を定める
    def eps_greed(self):
        if np.random.uniform() < self.eps:
            #探索状態
            i_act = np.random.randint(0,4)
        else:
            #搾取状態
            i_act = np.argmax(self.q_val[self.now_y][self.now_x])
        action = self.acts[i_act]
        self.prv_act=action
        return action

    def set_next_state(self,next_x,next_y,reward=0):
        #移動前の座標の更新
        self.prv_x = self.now_x
        self.prv_y = self.now_y
        #移動後の座標の更新
        self.now_x = next_x
        self.now_y = next_y
        #スタートでない場合行動価値関数の更新
        if reward !=0:
            self.reward.append(reward)
            self.q_update(reward)

    def q_update(self,reward):
        q = self.q_val[self.prv_y][self.prv_x][self.acts.index(self.prv_act)] #行動価値関数
        max_q = max(self.q_val[self.now_y][self.now_x]) #移動先の行動価値関数
        #移動先の行動価値関数によって行動価値関数をアップデート
        self.q_val[self.prv_y][self.prv_x][self.acts.index(self.prv_act)]= q + (self.alpha * (reward + (self.gamma * max_q) - q))


class Maze:
    def __init__(self):
        #迷路の設定
        #passable:0  impassable:-1 goal:1
        self.maze = [[-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
                     [-1, 0, 0,-1, 0, 0, 0, 0, 0, 0, 0,-1],
                     [-1, 0, 0,-1, 0,-1, 0, 0,-1,-1, 0,-1],
                     [-1,-1, 0,-1, 0,-1,-1,-1,-1, 0, 0,-1],
                     [-1, 0, 0, 0, 0, 0,-1, 0,-1, 0, 0,-1],
                     [-1, 0, 0, 0, 0, 0,-1, 0, 0, 0, 0,-1],
                     [-1,-1,-1,-1,-1, 0,-1, 0, 0, 0, 0,-1],
                     [-1, 0, 0, 0, 0, 0,-1, 0,-1, 0, 0,-1],
                     [-1, 0, 0,-1,-1,-1,-1, 0,-1, 0, 0,-1],
                     [-1, 0, 0, 0, 0,-1, 0, 0,-1, 0, 0,-1],
                     [-1, 0, 0,-1, 0, 0, 0, 0,-1, 0, 1,-1],
                     [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]]
        #行動の種類とスタート位置の設定
        self.action =['up','down','left','right']
        self.start_x = 1
        self.start_y = 1
        self.goal_x = 10
        self.goal_y = 10
        self.size_x = 12
        self.size_y = 12

    def start(self):
        return self.start_x,self.start_y

    def step(self,action,x,y):
        next_x = x
        next_y = y
        print("(x,y) = ({0}, {1})".format(x, y))

        reward = -1
        #goal = False
        #移動可能か否か調べる
        if action == 'up':
            next_y -= 1
            if self.maze[next_y][next_x] == -1:
                next_y += 1
        elif action == 'down':
            next_y += 1
            if self.maze[next_y][next_x] == -1:
                next_y -= 1
        elif action == 'left':
            next_x -= 1
            if self.maze[next_y][next_x] == -1:
                next_x += 1
        elif action == 'right':
            next_x += 1
            if self.maze[next_y][next_x] == -1:
                next_x -= 1

        #ゴールか否か
        if self.maze[next_y][next_x] == 1:
            print("Goal")
            return next_x, next_y, reward, True
        else:
            return next_x, next_y, reward, False

    def draw_all_maze(self,sc,x_ag,y_ag):
        for y_mz in range(self.size_y):
            for x_mz in range(self.size_x):
                X_mz = x_mz * self.size_x
                Y_mz = y_mz * self.size_y
                if self.maze[y_mz][x_mz] == -1:  # 壁
                    pygame.draw.rect(sc, GRAY, [X_mz, Y_mz, self.size_x, self.size_y])
                if self.maze[y_mz][x_mz] == 0:  # 通路
                    pygame.draw.rect(sc, WHITE, [X_mz, Y_mz, self.size_x, self.size_y])

        X_ag = x_ag * self.size_x
        Y_ag = y_ag * self.size_y
        pygame.draw.rect(sc, RED, [X_ag, Y_ag, self.size_x, self.size_y])

        # スタート
        X_st = self.start_x * self.size_x
        Y_st = self.start_y * self.size_y
        pygame.draw.rect(sc, BLUE, [X_st, Y_st, self.size_x, self.size_y])

        # ゴール
        x = self.goal_x
        y = self.goal_y
        X = self.goal_x * self.size_x
        Y = self.goal_y * self.size_y
        pygame.draw.rect(sc, BLUE, [X, Y, self.size_x, self.size_y])


EPISODE_NUM=1000
EPSILON=0.1
ALPHA=0.1
GAMMA=0.9
ACTIONS=['up','down','left','right']

def main():
    #pygameのインポート
    pygame.init()
    pygame.display.set_caption("Q-Learning")
    screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE + 20))
    clock = pygame.time.Clock()

    maze = Maze()
    x = maze.start_x
    y = maze.start_y
    agent = Agent(alpha=ALPHA, gamma=GAMMA, eps=EPSILON, acts=ACTIONS, now_x=x, now_y=y)
    rewards = []

    for episode in range(EPISODE_NUM):
        episode_rewards = []
        while (1):
            screen.fill(WHITE)
            # 迷路の描画
            maze.draw_all_maze(sc=screen,x_ag=x,y_ag=y)
            pygame.display.update()
            clock.tick(30)#プログラムのフレーム数

            action = agent.eps_greed()  # 行動を決める
            x, y, reward, goal = maze.step(action, x, y)  # mazeで行動後の座標と報酬
            agent.set_next_state(x, y, reward)
            episode_rewards.append(reward)
            # Goal or not
            if goal:
                break
        rewards.append(np.sum(episode_rewards))
        x, y = maze.start()
        agent.set_next_state(x, y)

    #グラフ壁画
    plt.plot(np.arange(EPISODE_NUM), rewards)
    plt.xlabel("episode")
    plt.ylabel("reward")
    plt.show()

'''
#GUIを表示しない場合
def main():
    maze = Maze()
    x=maze.start_x
    y=maze.start_y
    agent=Agent(alpha=ALPHA,gamma=GAMMA,eps=EPSILON,acts=ACTIONS,now_x=x,now_y=y)
    rewards=[]

    for episode in range(EPISODE_NUM):
        episode_rewards=[]
        while(1):
            action=agent.eps_greed() #行動を決める
            x,y,reward,goal= maze.step(action,x,y) #mazeで行動後の座標と報酬
            agent.set_next_state(x,y,reward)
            episode_rewards.append(reward)
            #Goal or not
            if goal:
                break
        rewards.append(np.sum(episode_rewards))
        x, y = maze.start()
        agent.set_next_state(x,y)

    #グラフ壁画
    plt.plot(np.arange(EPISODE_NUM), rewards)
    plt.xlabel("episode")
    plt.ylabel("reward")
    plt.show()
'''

if __name__ == '__main__':
    main()


