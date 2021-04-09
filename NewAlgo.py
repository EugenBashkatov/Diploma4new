from collections import deque as dq
import math
tst = dq()
tst1=dq()
tst.append([1,2,3,4,5])
tst.append([1,2,3])
tst1.append([10,20,30])
tst.append(tst1)
a=tst.pop()
a=tst.pop()
data_list = []
# data_list = dq()
data_invis = dq()
data_list.append(20.7)
data_list.append(17.9)
data_list.append(18.8)
data_list.append(14.6)
data_list.append(15.8)
data_list.append(15.8)
data_list.append(15.8)
data_list.append(17.4)
data_list.append(21.8)
data_list.append(20.0)
data_list.append(16.2)
data_list.append(13.3)
data_list.append(16.7)
data_list.append(21.5)

def get_k(x0, x1):
    global data_list
    y0 = data_list[x0]
    y1 = data_list[x1]

    k = (y1 - y0) / (x1 - x0)
    # if is_visible():
    return k

def is_visible():
    global stack_vis
    global stack_invis
    global data_list
    global test_k
    global step

    cur_dot = stack_vis.popleft()
    stack_vis.appendleft(cur_dot)



    x0=cur_dot[0]
    cur_k = cur_dot[1]
    x1= x0 + 1
    next_k = get_k(x0,x1)
    cur_k=test_k.popleft()
    test_k.appendleft(cur_k)
    return (next_k > cur_k)

def get_clusters():
    global stack_vis
    global stack_invis
    global data_list
    # global stack_invis
    len_stack=stack_vis.__len__()
    cur_dot = stack_vis[0]
    stack_invis.appendleft(cur_dot)
    x0=cur_dot[0]
    cur_k = cur_dot[1]
    x1= x0 + 1
    next_k = get_k(x0,x1)
    next_dot=[x1,next_k,data_list[x1]]
    vis=is_visible()
    print(vis)
    #print(stack_vis.index([1,2.,3],0,stack_vis.__len__()-1))

    if is_visible():
        #stack_vis.insert(1,[1,2.,3])
        stack_vis.appendleft(next_dot)
        #stack_vis.appendleft(stack_invis)
        print(stack_vis.__len__())
        stack_vis.popleft()
        test_k.appendleft(next_dot[1])
        stack_invis.popleft()
        print('1   vis={}'.format(stack_vis))
        print('1 invis={}'.format(stack_invis))

        get_clusters()
        exit()
    else:
        stack_invis.popleft()
        stack_invis.append(next_dot)
        print('2   vis={}'.format(stack_vis))
        print('2 invis={}'.format(stack_invis))


        get_clusters()
    return next_dot


if __name__ == '__main__':
    print("Hello python")
# ------------------------------------------
d_cluster_chain = {0: [0]}
d_cluster_length = {0: 1}
stack_vis = dq()
# stack_vis.append({0:0})
# stack_vis.append({0:1})
pointer=dq()
stack_invis = dq()
test_k=dq()

stack_vis.append([0,-math.inf,data_list[0]])
#stack_invis.append([0,-math.inf,data_list[0]])
test_k.appendleft(-math.inf)
get_clusters()

print('3   vis={}'.format(stack_vis.__len__()))
print('3 invis={}'.format(stack_invis))
print('4   vis={}'.format(stack_vis))
print('4 invis={}'.format(stack_invis))

# ------------------------------------------

