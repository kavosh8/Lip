import numpy, sys

def rewardToReturn(rewards,gamma):# takes a list of reward per t and converts it to return per t
    T=len(rewards)
    returns=T*[0]
    returns[T-1]=rewards[T-1] 
    for t in range(T-2,-1,-1):
        returns[t]=rewards[t]+gamma*returns[t+1]
    return returns

def rep_2_rep_and_action(rep,action,actionSize):
        rep_and_action=numpy.zeros((1,rep.shape[1]+actionSize))
        rep_and_action[0,0:rep.shape[1]]=rep
        one_hot_action=numpy.zeros((1,actionSize))
        one_hot_action[0,action]=1
        rep_and_action[0,rep.shape[1]:rep.shape[1]+actionSize]=one_hot_action
        return rep_and_action

def printLog(episode,info,batch_episode_number,frequency):
    if (episode) % frequency ==0:
        print("***")
        print("episode number",episode)
        print([a[0] for a,b,c,d in info[-50:]])
        average_return=numpy.mean([a[0] for a,b,c,d in info[-batch_episode_number:]])
        print("average return",average_return)
        print("***")
        sys.stdout.flush()

def save_stuff(actor,info,episode,run,batch_episode_number):
    if (episode) % batch_episode_number ==0:
        numpy.savetxt(str(run)+"-"+str(episode)+".txt",[numpy.mean([a[0] for a,b,c,d in info[-batch_episode_number:]])])
        actor.model.save_weights(str(run)+"-"+str(episode)+".h5")

def actions21hot(actions_list,action_size):
    out=[]
    for a in actions_list:
        temp=numpy.zeros(action_size)
        temp[a]=1
        out.append(temp)
    return out