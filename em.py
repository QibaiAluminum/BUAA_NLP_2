# 导入numpy库和random库
import numpy as np
import random

# EM算法过程函数定义
def em(data, esti_p, max_iter, eps=1e-3):
    '''
    输入：
    data：观测数据
    esti_p：初始化的估计参数值
    max_iter：最大迭代次数
    eps：收敛阈值
    输出：
    esti_p：估计参数
    '''
    # 初始化似然函数值
    llh_old = -np.infty
    for i in range(max_iter):
        # E步：求隐变量分布
        # 对数似然 [coin_num, exp_num]
        log_like = np.array([np.sum(data * np.log(theta), axis=1) for theta in esti_p])
        # 似然 [coin_num, exp_num]
        like = np.exp(log_like)
        # 求隐变量分布 [coin_num, exp_num]
        ws = like/like.sum(0)
        # 概率加权
        vs = np.array([w[:, None] * data for w in ws])
        # M步：更新参数值
        esti_p = np.array([v.sum(0)/v.sum() for v in vs])
        # 更新似然函数
        llh_new = np.sum([w*l for w, l in zip(ws, log_like)])
        print("Iteration: %d" % (i+1))
        print("A = %.2f,B = %.2f, C = %.2f, likelihood = %.2f"
              % (esti_p[0,0], esti_p[1,0],esti_p[2,0], llh_new))
        # 满足迭代条件即退出迭代
        if np.abs(llh_new - llh_old) < eps:
            break
        llh_old = llh_new
    return esti_p
# 生成随机投掷的函数
def random_pick(some_list, probabilities):
    x = random.uniform(0,1)
    cumulative_probability = 0.0
    for item, item_probability in zip(some_list, probabilities):
         cumulative_probability += item_probability
         if x < cumulative_probability:
               break
    return item

if __name__ == "__main__":
    # coin_list：硬币列表，0代表硬币a，1代表硬币b，2代表硬币c
    coin_list = [0, 1, 2]
    # probabilities：代表硬币混合比例，s1=0.3，s2=0.4,1-s1-s2=0.3
    probabilities = [0.3, 0.4, 0.3]
    # coin_ht:代表硬币的正反面，0代表反面，1代表正面
    coin_ht = [0, 1]
    # coin_single_pro:代表硬币正反面朝上的概率模型；硬币a正面朝上的概率0.7，硬币b正面朝上的概率为0.5，硬币c正面朝上的概率为0.3
    coin_single_pro = [[0.3, 0.7], [0.5, 0.5], [0.7, 0.3]]
    # observed:代表n组扔硬币，每组挑一个一个硬币扔十次，共计10n次；这里我选取扔50组，共计500次投掷
    observed=[]
    # coin:代表每组挑出的硬币类型，0、1、2分别对应硬币a，硬币b，硬币c
    coin = []
    # coin_dic：代表投掷次数的所有结果，以01表示，0代表反面，1代表正面
    coin_dic = []

    # print(random_pick(coin_list,probabilities))

    for i in range(0, 50):
        coin.append(random_pick(coin_list, probabilities))
    # print(coin)

    # print(len(coin))
    for i in range(0, len(coin)):
        coin_single = []
        for j in range(0, 10):
            # 硬币的概率coin_sin_pro[coin_list[i]]
            coin_single.append(random_pick(coin_ht, coin_single_pro[coin[i]]))
            # print(coin_single)
        coin_dic.append(coin_single)
    #print(coin_dic)

    for i in range(0, 50):
        num = 0
        for j in range(0, 10):
            num = num + coin_dic[i][j]
        observed.append((num, 10 - num))
        #print(num)
    print("Coin toss list：")
    print(coin_dic)
    # 观测数据，n次独立试验，每次试验10次抛掷的正反次数
    # 比如第一次试验为5次正面5次反面
    observed_data = np.array(observed)
    # 初始化参数值，这是一个猜测值，在这里我假设硬币a的正面概率为0.8，硬币B的正面概率为0.5，硬币C的正面概率为0.4
    esti_p = np.array([[0.8, 0.2], [0.5, 0.5],[0.4, 0.6]])
    esti_p = em(observed_data, esti_p, max_iter=500, eps=1e-4)
    print(esti_p)
